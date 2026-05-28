"""
sg_weekly_update.py
-------------------
Script hebdomadaire d'extraction SG Prospectus.

Usage:
    python sg_weekly_update.py

Variables d'environnement (toutes optionnelles) :
    SG_BASE_URL      URL de base de l'API  (défaut: https://prospectus.socgen.com)
    SG_USER_AGENT    User-Agent HTTP        (défaut: sg-extract/1.0)
    SG_AUTHORIZATION En-tête Authorization  (défaut: non envoyé)
    SG_API_KEY       En-tête x-api-key      (défaut: non envoyé)
    SG_COOKIE        En-tête Cookie         (défaut: non envoyé)
    SG_PAGE_SIZE     Taille d'une page API  (défaut: 5000)
    SG_MAX_PAGES     Nb max de pages/pays   (défaut: 500)
    SG_HTTP_RETRIES  Nb de tentatives HTTP  (défaut: 3)
    SG_COUNTRIES     Liste pays séparés par des virgules
                     (défaut: la liste complète ci-dessous)
    SG_DB_PATH       Chemin du fichier parquet base de données
                     (défaut: sg_database.parquet)
"""

import os
import sys
import time
import logging
from datetime import date

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL   = os.environ.get("SG_BASE_URL",   "https://prospectus.socgen.com")
PAGE_SIZE  = int(os.environ.get("SG_PAGE_SIZE",  "5000"))
MAX_PAGES  = int(os.environ.get("SG_MAX_PAGES",  "500"))
HTTP_RETRIES = int(os.environ.get("SG_HTTP_RETRIES", "3"))
DB_PATH    = os.environ.get("SG_DB_PATH", "data/data_SG.parquet")
SORT       = "document,ASC"

# Colonne remplie par sg_download_pdfs.py ; présente pour les nouvelles lignes (vide).
PDF_LOCAL_COL = "pdf_local_path"

COUNTRIES_DEFAULT = [
    "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Chile",
    "Colombia", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Finland",
    "France", "Germany", "Hong Kong", "Hungary", "Ireland", "Italy", "Japan",
    "Liechtenstein", "Luxembourg", "Malta", "Mexico", "Netherlands", "Norway",
    "Panama", "Peru", "Poland", "Portugal", "Romania", "Singapore", "Slovakia",
    "South Africa", "Spain", "Sweden", "Switzerland", "United Kingdom",
    "United States", "Uruguay",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_headers() -> dict:
    headers = {
        "Accept":     "application/json",
        "User-Agent": os.environ.get("SG_USER_AGENT", "sg-extract/1.0"),
    }
    if auth := os.environ.get("SG_AUTHORIZATION"):
        headers["Authorization"] = auth
    if key := os.environ.get("SG_API_KEY"):
        headers["x-api-key"] = key
    if cookie := os.environ.get("SG_COOKIE"):
        headers["Cookie"] = cookie
    return headers


def fetch_country(session: requests.Session, country: str) -> pd.DataFrame:
    """Télécharge toutes les pages d'un pays et retourne un DataFrame."""
    url = f"{BASE_URL.rstrip('/')}/api/v1/documents/product-documentations/metadata"
    df_country = pd.DataFrame()
    last_df    = pd.DataFrame()

    for page in range(MAX_PAGES):
        params = {"size": PAGE_SIZE, "page": page, "sort": SORT, "country": country}

        for attempt in range(1, HTTP_RETRIES + 1):
            try:
                resp = session.get(url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as exc:
                if attempt >= HTTP_RETRIES:
                    log.error("  Impossible d'accéder à l'API après %d tentatives : %s", HTTP_RETRIES, exc)
                    raise
                wait = 2 ** (attempt - 1)
                log.warning("  Erreur réseau (tentative %d/%d). Retry dans %ds…", attempt, HTTP_RETRIES, wait)
                time.sleep(wait)

        rows = data.get("responseList") or []
        if not rows:
            break

        df_new = pd.json_normalize(rows, sep="_")

        # Détection de boucle infinie (API renvoie la même page en boucle)
        if page > 0 and not last_df.empty and df_new.equals(last_df):
            break

        df_country = pd.concat([df_country, df_new], ignore_index=True)
        last_df    = df_new.copy()

        log.info("  %s – page %d : %d docs cumulés", country, page + 1, len(df_country))

    return df_country


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les transformations de mise en forme."""
    if df.empty:
        return df

    if "lastModificationDate" in df.columns:
        df["lastModificationDate"] = pd.to_datetime(
            df["lastModificationDate"], errors="coerce"
        ).dt.date

    for col in ("publicOfferPlace", "listingPlace"):
        if col in df.columns:
            mask = ~df[col].isin(["['_Unlisted_']", "['_No Public offer_']"])
            df.loc[mask, col] = df.loc[mask, col].apply(
                lambda x: str(x)[2:-2] if isinstance(x, str) and len(str(x)) > 4 else x
            )

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.drop_duplicates()
    return add_internet_link(df)


LEGALDOC_SEARCH_PREFIX = "https://prospectus.socgen.com/legaldoc_search/"


def add_internet_link(df: pd.DataFrame) -> pd.DataFrame:
    """URL stable vers la fiche document (isin + docid). Toujours ajoutée si les colonnes existent."""
    if df.empty:
        return df
    if "isin" not in df.columns or "docid" not in df.columns:
        log.warning("Colonnes isin ou docid absentes – internet_link non calculée.")
        return df
    isin = df["isin"].fillna("").astype(str).replace({"nan": "", "<NA>": ""})
    docid = df["docid"].fillna("").astype(str).replace({"nan": "", "<NA>": ""})
    df = df.copy()
    df["internet_link"] = LEGALDOC_SEARCH_PREFIX + isin + "/" + docid
    return df


def load_database(path: str) -> pd.DataFrame:
    """Charge la base existante ou retourne un DataFrame vide."""
    if os.path.exists(path):
        log.info("Base existante chargée : %s", path)
        return pd.read_parquet(path)
    log.info("Aucune base existante trouvée – une nouvelle sera créée : %s", path)
    return pd.DataFrame()


def save_database(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path, index=False)
    log.info("Base sauvegardée : %s  (%d lignes au total)", path, len(df))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    today = date.today()
    log.info("=== Démarrage de l'extraction SG – %s ===", today)

    # --- Construction des pays ---
    if env_countries := os.environ.get("SG_COUNTRIES"):
        countries = [c.strip() for c in env_countries.split(",") if c.strip()]
    else:
        countries = COUNTRIES_DEFAULT

    countries_api = [c.replace(" ", "_").upper() for c in countries]

    # --- Chargement de la base existante ---
    db = load_database(DB_PATH)
    if not db.empty and PDF_LOCAL_COL not in db.columns:
        db[PDF_LOCAL_COL] = pd.NA
    had_internet_link = "internet_link" in db.columns and len(db) > 0
    existing_ids: set = set(db["docid"].dropna()) if "docid" in db.columns else set()
    log.info("Nombre de documents déjà connus : %d", len(existing_ids))

    # --- Session HTTP ---
    session = requests.Session()
    session.headers.update(build_headers())

    # --- Extraction pays par pays ---
    frames_new: list[pd.DataFrame] = []
    total_fetched = 0

    for i, (country_raw, country_api) in enumerate(zip(countries, countries_api), start=1):
        log.info("[%d/%d] Extraction : %s", i, len(countries), country_raw)
        try:
            df_c = fetch_country(session, country_api)
        except requests.exceptions.RequestException:
            log.error("  Pays %s ignoré suite à une erreur réseau.", country_raw)
            continue

        df_c = format_dataframe(df_c)
        total_fetched += len(df_c)

        if df_c.empty or "docid" not in df_c.columns:
            log.info("  %s : aucune donnée.", country_raw)
            continue

        # Filtrage : uniquement les lignes dont le docid est inconnu
        df_new = df_c[~df_c["docid"].isin(existing_ids)].copy()

        if df_new.empty:
            log.info("  %s : 0 nouvelle ligne (tout était déjà en base).", country_raw)
            continue

        # Horodatage des nouvelles lignes
        df_new["date_added"] = today

        frames_new.append(df_new)

        # On met à jour l'ensemble des IDs connus pour éviter les doublons inter-pays
        existing_ids.update(df_new["docid"].dropna())

        log.info("  %s : +%d nouvelles lignes.", country_raw, len(df_new))

    # --- Consolidation & sauvegarde ---
    if not frames_new:
        log.info("Aucune nouvelle ligne à ajouter.")
        db = add_internet_link(db)
        if not db.empty and not had_internet_link:
            save_database(db, DB_PATH)
            log.info("Colonne internet_link ajoutée à la base existante.")
    else:
        df_added = pd.concat(frames_new, ignore_index=True)
        log.info("Nouvelles lignes à ajouter : %d", len(df_added))

        # Ajout de la colonne date_added dans la base existante si elle n'existe pas encore
        if not db.empty and "date_added" not in db.columns:
            db["date_added"] = pd.NaT

        db = pd.concat([db, df_added], ignore_index=True)
        if PDF_LOCAL_COL not in db.columns:
            db[PDF_LOCAL_COL] = pd.NA
        db = add_internet_link(db)
        save_database(db, DB_PATH)

    log.info(
        "=== Terminé. Docs récupérés ce run : %d | Docs en base : %d ===",
        total_fetched,
        len(db),
    )


if __name__ == "__main__":
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        log.error("pyarrow est requis pour lire/écrire des fichiers Parquet.")
        log.error("Installe-le avec :  pip install pyarrow")
        sys.exit(1)

    main()
