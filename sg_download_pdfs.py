"""
sg_download_pdfs.py
-------------------
Télécharge les PDFs listés dans le fichier parquet (défaut : data_SG.parquet),
uniquement pour les lignes sans chemin valide enregistré dans la colonne
`pdf_local_path` (chemin absolu du fichier sur disque).
Les lignes déjà marquées avec un fichier existant ne sont pas re-téléchargées.

Nommage des fichiers : {ISIN}_{TYPE}_{YYYYMMDD}.pdf
  - FINAL_TERMS              → FT
  - STANDALONE               → STANDALONE
  - NOTICE_TO_THE_NOTEHOLDERS→ PP
  - SIMPLIFIED_PROSPECTUS    → PP

Usage :
    python sg_download_pdfs.py

Variables d'environnement (toutes optionnelles) :
    SG_BASE_URL      URL de base  (défaut: https://prospectus.socgen.com)
    SG_USER_AGENT    User-Agent   (défaut: sg-extract/1.0)
    SG_AUTHORIZATION En-tête Authorization
    SG_API_KEY       En-tête x-api-key
    SG_COOKIE        En-tête Cookie
    SG_DB_PATH       Fichier parquet source (défaut: data_SG.parquet)
    SG_PDF_DIR       Dossier de destination (défaut: pdf_SG)
    SG_HTTP_RETRIES  Nb de tentatives HTTP  (défaut: 3)
    SG_MAX_DOWNLOADS Limite optionnelle du nombre de PDFs ce run (tests / débogage)
"""

import glob
import logging
import os
import re
import sys
import time
from datetime import date
from typing import Optional

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
BASE_URL     = os.environ.get("SG_BASE_URL", "https://prospectus.socgen.com")
DB_PATH      = os.environ.get("SG_DB_PATH",  "data/data_SG.parquet")
PDF_DIR      = os.environ.get("SG_PDF_DIR",  "data/pdf_SG")
HTTP_RETRIES = int(os.environ.get("SG_HTTP_RETRIES", "3"))
_max_dl      = os.environ.get("SG_MAX_DOWNLOADS")
MAX_DOWNLOADS = int(_max_dl) if (_max_dl and _max_dl.strip()) else None

# Colonne Parquet : chemin absolu du PDF téléchargé (remplie par ce script).
PDF_LOCAL_COL = "pdf_local_path"

# Ancien fichier d’index (avant la colonne parquet). Utilisé une fois pour ne pas
# re-télécharger des docs déjà récupérés sans entrée en base — tu peux le supprimer
# une fois la colonne `pdf_local_path` renseignée (relance complète si besoin).
LEGACY_DOWNLOADED_INDEX = os.path.join(PDF_DIR, ".downloaded_docids.txt")

# Correspondance valeur API → abréviation dans le nom de fichier
DOC_TYPE_MAP: dict[str, str] = {
    "FINAL_TERMS":               "FT",
    "STANDALONE":                "STANDALONE",
    "NOTICE_TO_THE_NOTEHOLDERS": "PP",
    "SIMPLIFIED_PROSPECTUS":     "PP",
}

# Caractères interdits / dangereux dans un nom de fichier (séparateurs de chemin, etc.)
_INVALID_FS_CHUNK = re.compile(r"[/\\:\0\r\n]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_isin_for_filename(isin: object) -> str:
    """
    Nettoie l’ISIN pour un nom de fichier : pas de `/`, pas de sous-dossiers,
    multi-ISIN « A, B » → premier ISIN uniquement.
    """
    try:
        if isin is None or pd.isna(isin):
            return "UNKNOWN"
    except (ValueError, TypeError):
        return "UNKNOWN"
    s = str(isin).strip()
    if not s or s.lower() in ("nan", "<na>", "none"):
        return "UNKNOWN"
    if "," in s:
        s = s.split(",")[0].strip()
    s = _INVALID_FS_CHUNK.sub("_", s)
    s = re.sub(r"[\s]+", "_", s)
    s = _MULTI_UNDERSCORE.sub("_", s).strip("_")
    if not s:
        return "UNKNOWN"
    if len(s) > 120:
        s = s[:120]
    return s


def sanitize_doc_type_segment(doc_type_raw: object) -> str:
    """Segment type de document sans caractères réservés au FS."""
    raw = str(doc_type_raw).strip() if doc_type_raw is not None else ""
    abbr = DOC_TYPE_MAP.get(raw.upper(), raw)
    abbr = _INVALID_FS_CHUNK.sub("_", abbr)
    abbr = re.sub(r"[\s,]+", "_", abbr)
    abbr = _MULTI_UNDERSCORE.sub("_", abbr).strip("_")
    return abbr or "DOC"


def safe_to_parquet(df: pd.DataFrame, path: str) -> bool:
    """Écrit le parquet ; ne lève pas — retourne False en cas d’échec."""
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        log.exception("Écriture Parquet impossible (%s) : %s", path, exc)
        return False

def build_headers() -> dict:
    headers = {
        "Accept":     "application/pdf, application/octet-stream, */*",
        "User-Agent": os.environ.get("SG_USER_AGENT", "sg-extract/1.0"),
    }
    if auth := os.environ.get("SG_AUTHORIZATION"):
        headers["Authorization"] = auth
    if key := os.environ.get("SG_API_KEY"):
        headers["x-api-key"] = key
    if cookie := os.environ.get("SG_COOKIE"):
        headers["Cookie"] = cookie
    return headers


def load_legacy_downloaded_ids(index_path: str) -> set[str]:
    """Ancien index texte (rétrocompatibilité)."""
    if not os.path.exists(index_path):
        return set()
    with open(index_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def try_resolve_existing_pdf(isin: object, document_type: object) -> Optional[str]:
    """
    Si un PDF existe déjà dans pdf_SG avec le nom attendu {isin}_{abr}_*.pdf,
    retourne son chemin absolu (le plus récent si plusieurs).
    """
    isin_safe = sanitize_isin_for_filename(isin)
    doc_key = str(document_type).strip().upper()
    abbr = DOC_TYPE_MAP.get(doc_key)
    if not abbr:
        return None
    pattern = os.path.join(PDF_DIR, f"{isin_safe}_{abbr}_*.pdf")
    matches = glob.glob(pattern)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return os.path.abspath(matches[0])


def cell_has_valid_pdf_file(val) -> bool:
    """True si la cellule contient un chemin vers un fichier PDF existant."""
    try:
        if val is None or pd.isna(val):
            return False
    except (ValueError, TypeError):
        return False
    s = str(val).strip()
    if not s or s.lower() in ("nan", "<na>", "none"):
        return False
    p = os.path.abspath(s) if not os.path.isabs(s) else s
    return os.path.isfile(p)


def make_filename(isin: str, doc_type_raw: str, download_date: date) -> str:
    """Construit le nom de fichier selon la convention définie (sans chemin réservé)."""
    isin_clean = sanitize_isin_for_filename(isin)
    type_abbr = sanitize_doc_type_segment(doc_type_raw)
    date_str = download_date.strftime("%Y%m%d")
    name = f"{isin_clean}_{type_abbr}_{date_str}.pdf"
    return os.path.basename(name)


def build_pdf_api_url(docid: str) -> str:
    """
    URL réelle du flux PDF (l’URL relative links_content pointe vers la SPA HTML).
    """
    from urllib.parse import quote

    base = BASE_URL.rstrip("/")
    mid = "/api/v1/documents/product-documentations/contents"
    mt = quote("application/pdf", safe="")
    return f"{base}{mid}?docId={docid}&mimeType={mt}"


def unique_path(folder: str, filename: str) -> str:
    """
    Retourne un chemin unique : si le fichier existe déjà (même isin/type/date
    mais docid différent), ajoute un suffixe _2, _3, …
    """
    filename = os.path.basename(filename)
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{base}_{counter}{ext}")
        counter += 1
    return candidate


def download_pdf(session: requests.Session, docid: str, dest_path: str) -> bool:
    """
    Télécharge le PDF depuis l'API REST et le sauvegarde.
    Retourne True en cas de succès, False sinon (ne lève pas d'exception).
    """
    dest_path = os.path.join(os.path.dirname(dest_path) or ".", os.path.basename(dest_path))
    url = build_pdf_api_url(docid)

    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = session.get(url, timeout=120, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                log.warning("    Réponse HTML reçue (authentification requise ?) – skip.")
                return False

            # Accumuler d'abord les premiers octets pour valider le PDF
            first = next(resp.iter_content(chunk_size=65536), b"")
            if not first.startswith(b"%PDF"):
                log.warning("    Contenu inattendu (pas d'en-tête %%PDF) – skip.")
                return False

            try:
                os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(first)
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            except OSError as exc:
                log.exception("    Écriture disque impossible (%s) : %s", dest_path, exc)
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                except OSError:
                    pass
                return False
            return True

        except requests.exceptions.RequestException as exc:
            if attempt >= HTTP_RETRIES:
                log.error("    Échec après %d tentatives : %s", HTTP_RETRIES, exc)
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                except OSError:
                    pass
                return False
            wait = 2 ** (attempt - 1)
            log.warning("    Erreur réseau (tentative %d/%d). Retry dans %ds…", attempt, HTTP_RETRIES, wait)
            time.sleep(wait)
        except Exception as exc:
            log.exception("    Erreur inattendue pendant le téléchargement : %s", exc)
            try:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            except OSError:
                pass
            return False

    return False


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    today = date.today()
    log.info("=== Démarrage du téléchargement PDF SG – %s ===", today)

    try:
        # --- Vérifications préliminaires ---
        if not os.path.exists(DB_PATH):
            log.error("Fichier parquet introuvable : %s", DB_PATH)
            log.error("Lance d'abord sg_weekly_update.py pour construire la base.")
            return

        os.makedirs(PDF_DIR, exist_ok=True)

        # --- Chargement de la base ---
        log.info("Chargement du parquet : %s", DB_PATH)
        try:
            df = pd.read_parquet(DB_PATH)
        except Exception as exc:
            log.exception("Impossible de lire le parquet %s : %s", DB_PATH, exc)
            return

        schema_missing_pdf_col = PDF_LOCAL_COL not in df.columns
        log.info("Lignes dans la base : %d", len(df))

        required_cols = {"docid", "isin", "document"}
        missing = required_cols - set(df.columns)
        if missing:
            log.error("Colonnes manquantes dans le parquet : %s", missing)
            return

        df = df[df["docid"].notna() & (df["docid"].astype(str).str.len() > 0)].copy()
        log.info("Lignes avec docid utilisable : %d", len(df))

        if PDF_LOCAL_COL not in df.columns:
            df[PDF_LOCAL_COL] = pd.NA

        legacy_ids = load_legacy_downloaded_ids(LEGACY_DOWNLOADED_INDEX)
        if legacy_ids:
            log.info(
                "Ancien fichier %s : %d docids — tentative d’association avec des PDF déjà dans %s.",
                os.path.basename(LEGACY_DOWNLOADED_INDEX),
                len(legacy_ids),
                PDF_DIR,
            )
            backfilled = 0
            for idx in df.index:
                try:
                    if str(df.at[idx, "docid"]) not in legacy_ids:
                        continue
                    if cell_has_valid_pdf_file(df.at[idx, PDF_LOCAL_COL]):
                        continue
                    resolved = try_resolve_existing_pdf(df.at[idx, "isin"], df.at[idx, "document"])
                    if resolved:
                        df.at[idx, PDF_LOCAL_COL] = resolved
                        backfilled += 1
                except Exception as exc:
                    log.exception("Rétro-remplissage ignoré pour l’index %s : %s", idx, exc)
            if backfilled:
                if safe_to_parquet(df, DB_PATH):
                    log.info(
                        "Rétro-remplissage : %d chemins écrits depuis des fichiers déjà présents.",
                        backfilled,
                    )

        already_ok = df[PDF_LOCAL_COL].map(cell_has_valid_pdf_file)
        skipped_legacy = (
            ~already_ok
            & df["docid"].astype(str).isin(legacy_ids)
        )
        need_download = ~already_ok & ~skipped_legacy

        n_skip_ok = int(already_ok.sum())
        n_skip_legacy = int(skipped_legacy.sum())
        n_need = int(need_download.sum())
        log.info(
            "Déjà en base avec fichier présent : %d | Ignorés (index legacy) : %d | À télécharger : %d",
            n_skip_ok,
            n_skip_legacy,
            n_need,
        )

        idx_list = df.index[need_download].tolist()
        if MAX_DOWNLOADS is not None:
            idx_list = idx_list[:MAX_DOWNLOADS]
            log.info("Limite SG_MAX_DOWNLOADS=%s appliquée.", MAX_DOWNLOADS)

        if not idx_list:
            log.info("Rien à télécharger.")
            if schema_missing_pdf_col:
                if safe_to_parquet(df, DB_PATH):
                    log.info("Colonne %s ajoutée au fichier (valeurs vides).", PDF_LOCAL_COL)
            log.info(
                "=== Terminé (pas de téléchargement). Déjà OK : %d | Legacy sans chemin : %d ===",
                n_skip_ok,
                n_skip_legacy,
            )
            return

        log.info("PDFs à télécharger ce run : %d", len(idx_list))

        # --- Session HTTP ---
        session = requests.Session()
        session.headers.update(build_headers())

        # --- Téléchargements (réécriture parquet après chaque succès) ---
        success_count = 0
        error_count = 0

        for i, idx in enumerate(idx_list, start=1):
            try:
                row = df.loc[idx]
                docid = str(row["docid"])
                isin = str(row["isin"])
                doc_type_raw = str(row["document"])

                filename = make_filename(isin, doc_type_raw, today)
                dest_path = unique_path(PDF_DIR, filename)

                log.info(
                    "[%d/%d] %s  →  %s",
                    i,
                    len(idx_list),
                    docid[:8] + "…",
                    os.path.basename(dest_path),
                )

                ok = download_pdf(session, docid, dest_path)

                if ok:
                    abs_path = os.path.abspath(dest_path)
                    df.at[idx, PDF_LOCAL_COL] = abs_path
                    if safe_to_parquet(df, DB_PATH):
                        success_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1

            except Exception as exc:
                log.exception(
                    "Ligne ignorée après erreur (on continue) — index=%s : %s",
                    idx,
                    exc,
                )
                error_count += 1

            try:
                time.sleep(0.2)
            except Exception:
                pass

        log.info(
            "=== Terminé. Téléchargés : %d | Erreurs : %d | Déjà OK : %d | Legacy sans chemin : %d ===",
            success_count,
            error_count,
            n_skip_ok,
            n_skip_legacy,
        )
        log.info(
            "PDFs dans : %s — colonne %s mise à jour dans %s",
            os.path.abspath(PDF_DIR),
            PDF_LOCAL_COL,
            DB_PATH,
        )

    except Exception as exc:
        log.exception("Erreur globale dans main() (fin contrôlée) : %s", exc)


if __name__ == "__main__":
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        log.error("pyarrow est requis pour lire le fichier Parquet.")
        log.error("Installe-le avec :  pip install pyarrow")
        sys.exit(1)

    try:
        main()
    except Exception:
        log.exception("Erreur non gérée au lancement — le processus se termine sans lever.")
