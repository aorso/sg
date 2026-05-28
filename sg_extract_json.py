"""
sg_extract_json.py
------------------
Convertit les PDFs de term sheets / final terms en JSON structuré via l'API Claude.

Mode synchrone (défaut) : 1 appel à la fois, résultat immédiat.
  → Utiliser pour la validation sur les 10 premiers docs (SG_MAX_EXTRACTIONS=10).

Mode batch (à venir) : prévu section 8 — basculer quand le prompt est validé.

Convention de nommage JSON : même base que le fichier PDF, extension .json
  FR001400I7D4_FT_20260502.pdf → json_SG/FR001400I7D4_FT_20260502.json

Usage :
    export ANTHROPIC_API_KEY=sk-ant-...
    python sg_extract_json.py

    # Limiter à 10 docs pour validation
    SG_MAX_EXTRACTIONS=10 python sg_extract_json.py

    # Filtrer un ISIN spécifique (debug)
    SG_FILTER_ISIN=FR001400I7D4 python sg_extract_json.py

Variables d'environnement :
    ANTHROPIC_API_KEY    Clé API Anthropic (requise)
    SG_DB_PATH           Fichier parquet source   (défaut: data_SG.parquet)
    SG_PDF_DIR           Dossier des PDFs         (défaut: pdf_SG)
    SG_JSON_DIR          Dossier de sortie JSON   (défaut: json_SG)
    SG_PROMPT_PATH       Chemin du prompt         (défaut: extraction_prompt.txt)
    SG_MODEL             Modèle Claude            (défaut: claude-haiku-4-5-20251001)
    SG_MAX_EXTRACTIONS   Limite optionnelle       (tests / validation)
    SG_FILTER_ISIN       Traiter uniquement cet ISIN (debug)
"""

import base64
import json
import logging
import os
import sys
import time
from datetime import date
from typing import Optional

import anthropic
import pandas as pd

# ---------------------------------------------------------------------------
# Chargement optionnel d'un fichier .env (sans dépendance externe)
# ---------------------------------------------------------------------------
def load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Charge des variables d'environnement depuis un fichier .env local.
    Format supporté (minimum) :
      KEY=value
      KEY="value with spaces"
    Les lignes vides et celles commençant par # sont ignorées.
    Ne remplace pas les variables déjà définies dans l'environnement.
    """
    try:
        if not os.path.exists(dotenv_path):
            return
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Échec silencieux : on garde la configuration via l'environnement
        return


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
DB_PATH     = os.environ.get("SG_DB_PATH",    "data/data_SG.parquet")
PDF_DIR     = os.environ.get("SG_PDF_DIR",    "data/pdf_SG")
JSON_DIR    = os.environ.get("SG_JSON_DIR",   "data/json_SG")
PROMPT_PATH = os.environ.get("SG_PROMPT_PATH", "config/extraction_prompt.txt")
MODEL       = os.environ.get("SG_MODEL",      "claude-haiku-4-5-20251001")
FILTER_ISIN = os.environ.get("SG_FILTER_ISIN", "").strip()
_max_ex     = os.environ.get("SG_MAX_EXTRACTIONS", "").strip()
MAX_EXTRACTIONS = int(_max_ex) if _max_ex else None

PDF_LOCAL_COL  = "pdf_local_path"
JSON_LOCAL_COL = "json_local_path"

# Pause entre deux appels API synchrones (respecter les rate limits)
INTER_CALL_SLEEP = 0.5


# ---------------------------------------------------------------------------
# max_tokens adaptatif selon le nombre de pages
#
# Densité observée : ~1 500–2 000 tokens JSON/page (tableaux financiers inclus).
# On applique une marge de sécurité ×1.5 par rapport au pire cas.
# max_tokens est un PLAFOND : si la réponse est plus courte, aucun surcoût.
#
# Plafond absolu : SG_MAX_OUTPUT_TOKENS (défaut 64 000).
# Si le modèle a une limite inférieure (ex: 32 000), définir :
#   export SG_MAX_OUTPUT_TOKENS=32000
# ---------------------------------------------------------------------------
# Plafond absolu du modèle. Haiku 4.5 : au moins 32 000 (ajuster si erreur 400).
# Inclut le budget de thinking interne du modèle + le JSON de sortie.
_MODEL_OUTPUT_CEILING = int(os.environ.get("SG_MAX_OUTPUT_TOKENS", "32000"))


def max_tokens_for_pages(n_pages: int) -> int:
    # Haiku 4.5 utilise du thinking interne dont les tokens sont décomptés
    # du même budget max_tokens que la réponse texte.
    # Observation : 12p → 12 000 tokens utilisés SANS produire de JSON (thinking pur).
    # Estimation conservative : 10 000 tok de thinking + 1 500 tok/page de JSON.
    # Marge ×1.5 pour éviter toute troncature.
    tokens = max(20_000, int((10_000 + n_pages * 1_500) * 1.5))
    return min(tokens, _MODEL_OUTPUT_CEILING)


# ---------------------------------------------------------------------------
# Helpers PDF
# ---------------------------------------------------------------------------
def count_pdf_pages(path: str) -> int:
    """Retourne le nombre de pages du PDF. Fallback conservateur : 15."""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)
    except Exception as exc:
        log.warning("    Impossible de compter les pages (%s) — fallback 15", exc)
        return 15


def is_scanned_pdf(path: str) -> bool:
    """
    Heuristique : un PDF scanné produit peu ou pas de texte extractible.
    Si < 100 caractères sur les 3 premières pages → probablement scanné.
    """
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:3]:
                text = page.extract_text() or ""
                if len(text.strip()) > 100:
                    return False
        return True
    except Exception:
        return False


def pdf_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Helpers parquet
# ---------------------------------------------------------------------------
def cell_has_valid_json_file(val) -> bool:
    """True si la cellule contient un chemin vers un fichier JSON existant."""
    try:
        if val is None or pd.isna(val):
            return False
    except (ValueError, TypeError):
        return False
    s = str(val).strip()
    if not s or s.lower() in ("nan", "<na>", "none"):
        return False
    return os.path.isfile(s)


def cell_has_valid_pdf_file(val) -> bool:
    try:
        if val is None or pd.isna(val):
            return False
    except (ValueError, TypeError):
        return False
    s = str(val).strip()
    if not s or s.lower() in ("nan", "<na>", "none"):
        return False
    return os.path.isfile(s)


def safe_to_parquet(df: pd.DataFrame, path: str) -> bool:
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        log.exception("Écriture Parquet impossible (%s) : %s", path, exc)
        return False


def unique_path(folder: str, filename: str) -> str:
    """Retourne un chemin unique (ajoute _2, _3, … si nécessaire)."""
    base, ext = os.path.splitext(os.path.basename(filename))
    candidate = os.path.join(folder, filename)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{base}_{counter}{ext}")
        counter += 1
    return candidate


# ---------------------------------------------------------------------------
# Normalisation de la sortie modèle → JSON pur (sans fences Markdown)
# ---------------------------------------------------------------------------
def normalize_model_json_response(text: str) -> str:
    """
    Retire les artefacts courants autour du JSON : fences ```json … ```,
    préambule avant la première « { » ou « [ », BOM UTF-8.

    Idempotent : plusieurs passages ne dégradent pas le résultat.
    """
    if text is None:
        return ""
    s = text.strip()
    if not s:
        return ""

    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff").strip()

    # Bloc Markdown ``` ou ```json … ```
    if s.startswith("```"):
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.lstrip("\r\n")
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    if s.endswith("```"):
        s = s[:-3].strip()

    # Préambule éventuel avant la racine JSON
    br = s.find("{")
    sq = s.find("[")
    starts = [x for x in (br, sq) if x >= 0]
    if starts:
        left = min(starts)
        if left > 0:
            s = s[left:]

    return s.strip()


# ---------------------------------------------------------------------------
# Validation de la sortie JSON de Claude
# ---------------------------------------------------------------------------
def validate_json_output(raw: str, stop_reason: str) -> tuple[bool, Optional[dict], list[str]]:
    """
    Valide la sortie de Claude.
    Retourne (is_valid, parsed_dict, liste_de_warnings).

    Critères de validité :
      1. JSON parseable (après normalize_model_json_response)
      2. Contient _meta ou _index_recherche (structure attendue)
      3. stop_reason != 'max_tokens' (pas tronqué)
    """
    warnings: list[str] = []
    normalized = normalize_model_json_response(raw)

    # Vérifier la troncature via stop_reason (meilleure source que l'heuristique textuelle)
    if stop_reason == "max_tokens":
        warnings.append("Réponse tronquée (stop_reason=max_tokens) — JSON probablement incomplet")

    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as e:
        warnings.append(f"JSON non parseable : {e}")
        return False, None, warnings

    if not isinstance(parsed, dict):
        warnings.append("Racine JSON n'est pas un objet")
        return False, parsed, warnings

    if "_meta" not in parsed and "_search_index" not in parsed:
        warnings.append("Neither _meta nor _search_index found — unexpected structure")
        return False, parsed, warnings

    return True, parsed, warnings


# ---------------------------------------------------------------------------
# Appel API Claude (synchrone)
# ---------------------------------------------------------------------------
def extract_one(
    client: anthropic.Anthropic,
    system_prompt: str,
    pdf_path: str,
) -> tuple[bool, Optional[str], Optional[dict], str]:
    """
    Envoie un PDF à Claude et retourne (is_valid, raw_text, parsed_dict, stop_reason).
    Ne lève pas d'exception — retourne (False, None, None, "") en cas d'erreur.
    """
    n_pages = count_pdf_pages(pdf_path)
    mt = max_tokens_for_pages(n_pages)

    scanned = is_scanned_pdf(pdf_path)
    if scanned:
        log.warning("    PDF potentiellement scanné (peu de texte extractible)")

    log.info("    %d pages → max_tokens=%d%s", n_pages, mt, " [scanné?]" if scanned else "")

    try:
        pdf_b64 = pdf_to_base64(pdf_path)
    except OSError as exc:
        log.error("    Impossible de lire le PDF : %s", exc)
        return False, None, None, ""

    # Le SDK Anthropic exige le streaming quand max_tokens est élevé.
    # On utilise get_final_message() et on extrait uniquement les blocs "text"
    # (les ThinkingBlocks ont type="thinking" et sont ignorés).
    try:
        raw = ""
        stop_reason = ""
        usage = None
        with client.messages.stream(
            model=MODEL,
            max_tokens=mt,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Convert this financial document to JSON "
                                "following the schema defined in the system prompt. "
                                "Output only valid JSON, no markdown."
                            ),
                        },
                    ],
                }
            ],
        ) as stream:
            final_msg   = stream.get_final_message()
            stop_reason = final_msg.stop_reason or ""
            usage       = final_msg.usage

        # Journaliser les types de blocs pour le diagnostic
        block_types = [getattr(b, "type", "?") for b in (final_msg.content or [])]
        log.info("    content blocks: %s", block_types)

        # Extraire le texte des blocs "text" uniquement
        raw = "".join(
            getattr(b, "text", "") or ""
            for b in (final_msg.content or [])
            if getattr(b, "type", None) == "text"
        )

    except anthropic.AuthenticationError as exc:
        log.error("    Authentification Anthropic échouée : %s", exc)
        return False, None, None, ""
    except anthropic.RateLimitError as exc:
        log.error("    Rate limit atteint : %s", exc)
        return False, None, None, ""
    except anthropic.APIError as exc:
        log.error("    Erreur API Anthropic : %s", exc)
        return False, None, None, ""
    except Exception as exc:
        log.exception("    Erreur inattendue pendant l'appel API : %s", exc)
        return False, None, None, ""

    out_tokens = usage.output_tokens if usage else "?"
    log.info("    output_tokens=%s  stop=%s", out_tokens, stop_reason)

    is_valid, parsed, warnings = validate_json_output(raw, stop_reason)
    normalized = normalize_model_json_response(raw)
    if normalized != (raw or "").strip():
        log.info("    Sortie normalisée (fences Markdown / préambule retirés)")
    for w in warnings:
        log.warning("    %s", w)

    return is_valid, normalized, parsed, stop_reason


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> None:
    today = date.today()
    log.info("=== Démarrage de l'extraction JSON SG — %s ===", today)
    log.info("Modèle : %s", MODEL)

    # Charger un .env local si présent (optionnel)
    load_dotenv(".env")

    # --- Clé API ---
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        log.error("Variable d'environnement ANTHROPIC_API_KEY non définie.")
        log.error("Lance :  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # --- Prompt ---
    if not os.path.exists(PROMPT_PATH):
        log.error("Fichier prompt introuvable : %s", PROMPT_PATH)
        sys.exit(1)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    log.info("Prompt chargé : %s (%d caractères)", PROMPT_PATH, len(system_prompt))

    # --- Parquet ---
    if not os.path.exists(DB_PATH):
        log.error("Fichier parquet introuvable : %s", DB_PATH)
        sys.exit(1)
    try:
        df = pd.read_parquet(DB_PATH)
    except Exception as exc:
        log.exception("Impossible de lire le parquet : %s", exc)
        sys.exit(1)

    if JSON_LOCAL_COL not in df.columns:
        df[JSON_LOCAL_COL] = pd.NA

    os.makedirs(JSON_DIR, exist_ok=True)

    # --- Filtres ---
    has_pdf  = df[PDF_LOCAL_COL].map(cell_has_valid_pdf_file)
    has_json = df[JSON_LOCAL_COL].map(cell_has_valid_json_file)
    need_extract = has_pdf & ~has_json

    if FILTER_ISIN:
        need_extract = need_extract & (df["isin"].astype(str) == FILTER_ISIN)
        log.info("Filtre ISIN actif : %s", FILTER_ISIN)

    idx_list = df.index[need_extract].tolist()

    log.info(
        "PDFs disponibles : %d | Déjà extraits : %d | À extraire : %d",
        int(has_pdf.sum()),
        int(has_json.sum()),
        len(idx_list),
    )

    if MAX_EXTRACTIONS is not None:
        idx_list = idx_list[:MAX_EXTRACTIONS]
        log.info("Limite SG_MAX_EXTRACTIONS=%d appliquée.", MAX_EXTRACTIONS)

    if not idx_list:
        log.info("Rien à extraire.")
        return

    # --- Session Anthropic ---
    client = anthropic.Anthropic(api_key=api_key)

    success_count  = 0
    error_count    = 0
    truncated_count = 0

    for i, idx in enumerate(idx_list, start=1):
        try:
            row      = df.loc[idx]
            pdf_path = str(row[PDF_LOCAL_COL])
            isin     = str(row.get("isin", "UNKNOWN"))

            pdf_basename  = os.path.splitext(os.path.basename(pdf_path))[0]
            json_filename = f"{pdf_basename}.json"
            json_dest     = unique_path(JSON_DIR, json_filename)

            log.info("[%d/%d] %s  →  %s", i, len(idx_list), isin, os.path.basename(json_dest))

            is_valid, raw, parsed, stop_reason = extract_one(client, system_prompt, pdf_path)

            # --- Cas : sortie invalide mais texte récupéré → sauver pour debug ---
            if raw and not is_valid:
                error_path = json_dest.replace(".json", "_INVALID.json")
                try:
                    with open(error_path, "w", encoding="utf-8") as f:
                        f.write(raw)
                    log.warning(
                        "    Sortie invalide sauvegardée pour inspection : %s",
                        os.path.basename(error_path),
                    )
                except OSError:
                    pass
                error_count += 1
                continue

            # --- Cas : extraction valide ---
            if is_valid and parsed is not None:
                if stop_reason == "max_tokens":
                    parsed["_truncated"] = True
                    truncated_count += 1

                try:
                    with open(json_dest, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)
                    abs_path = os.path.abspath(json_dest)
                    df.at[idx, JSON_LOCAL_COL] = abs_path
                    if safe_to_parquet(df, DB_PATH):
                        success_count += 1
                        log.info("    ✓ Sauvegardé")
                    else:
                        error_count += 1
                except OSError as exc:
                    log.error("    Écriture JSON impossible : %s", exc)
                    error_count += 1
            else:
                error_count += 1

        except Exception as exc:
            log.exception("Ligne ignorée après erreur (on continue) — index=%s : %s", idx, exc)
            error_count += 1

        try:
            time.sleep(INTER_CALL_SLEEP)
        except Exception:
            pass

    # --- Résumé ---
    log.info(
        "=== Terminé. Succès : %d | Erreurs : %d | Tronqués : %d ===",
        success_count,
        error_count,
        truncated_count,
    )
    if truncated_count:
        log.warning(
            "%d doc(s) tronqué(s) — relancer avec SG_MAX_TOKENS plus élevé ou "
            "vérifier si le doc dépasse 30 pages.",
            truncated_count,
        )
    log.info("JSONs dans : %s", os.path.abspath(JSON_DIR))
    log.info("Colonne %s mise à jour dans %s", JSON_LOCAL_COL, DB_PATH)


# ---------------------------------------------------------------------------
# Vérification des dépendances au démarrage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    missing = []
    for pkg, install in [("pyarrow", "pyarrow"), ("anthropic", "anthropic"), ("pdfplumber", "pdfplumber")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install)

    if missing:
        log.error("Dépendances manquantes. Lance :  pip install %s", " ".join(missing))
        sys.exit(1)

    try:
        main()
    except Exception:
        log.exception("Erreur non gérée au lancement.")
