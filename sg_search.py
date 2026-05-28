"""
sg_search.py
------------
Outil de recherche dans les JSON SG.

Workflow type (notebook) :
    s = SGSearch(json_dir="json_SG", parquet_path="data_SG.parquet")
    s.build_index()

    labels = s.search_labels(r"retail|interdiction")
    display(labels)

    # Optionnel : voir les valeurs distinctes pour les labels choisis
    values = s.list_values(labels, selected_indices=[0, 1])
    display(values)

    isins  = s.filter(
        labels,
        selected_indices=[0, 1],
        values_df=values,
        selected_value_indices=[0],   # ne garder que certaines valeurs
    )
    df     = s.to_dataframe(isins)
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase + suppression accents + collapse whitespace."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Walker récursif
# ---------------------------------------------------------------------------

# Schéma JSON polymorphe (FR ⇄ EN) — un même rôle peut avoir plusieurs noms de clés.
_TYPE_ALIASES = {
    "section":      {"section"},
    "avertissement": {"avertissement", "warning"},
    "champ":        {"champ", "field"},
    "texte_libre":  {"texte_libre", "free_text"},
    "sous_section": {"sous_section", "subsection", "sub_section"},
    "tableau":      {"tableau", "table"},
}
_TYPE_LOOKUP = {alias: canonical for canonical, aliases in _TYPE_ALIASES.items() for alias in aliases}

_TITRE_KEYS    = ("_titre", "_title")
_CLE_KEYS      = ("_cle", "_key")
_VALEUR_KEYS   = ("_valeur", "_value")
_CONTENU_KEYS  = ("_contenu", "_content")
_CONTENT_LIST  = ("contenu", "content")


def _first(d: dict, keys, default=""):
    """Retourne la première valeur trouvée parmi les clés possibles."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _walk(node: dict, isin: str, json_file: str, path: str, records: list) -> None:
    """Parcourt récursivement un nœud JSON (schéma FR ou EN) et alimente records."""
    if not isinstance(node, dict):
        return

    raw_type = node.get("_type", "")
    ntype = _TYPE_LOOKUP.get(raw_type, raw_type)  # canonique

    if raw_type == "document":
        isin = node.get("_meta", {}).get("isin", isin)
        for i, section in enumerate(node.get("sections", [])):
            _walk(section, isin, json_file, f"sections[{i}]", records)
        return

    children_key = next((k for k in _CONTENT_LIST if k in node), None)
    children = node.get(children_key, []) if children_key else []

    if ntype in ("section", "avertissement"):
        label = _first(node, _TITRE_KEYS, "")
        content = _first(node, _CONTENU_KEYS, "") or ""
        # warnings EN ont souvent juste _content (pas de titre) → on indexe quand même
        if label or (ntype == "avertissement" and content):
            records.append({
                "isin": isin,
                "json_file": json_file,
                "json_path": path,
                "node_type": ntype,
                "label_original": label,
                "label": _normalize(label) if label else "",
                "value": str(content),
            })
        for i, child in enumerate(children):
            _walk(child, isin, json_file, f"{path}.{children_key}[{i}]", records)

    elif ntype == "champ":
        label = _first(node, _CLE_KEYS, "")
        value = _first(node, _VALEUR_KEYS, "") or ""
        if label:
            records.append({
                "isin": isin,
                "json_file": json_file,
                "json_path": path,
                "node_type": ntype,
                "label_original": label,
                "label": _normalize(label),
                "value": str(value),
            })
        # Sous-nœuds éventuels (ex: tableau_sous_jacents)
        skip = {"_type", *_CLE_KEYS, *_VALEUR_KEYS}
        for k, v in node.items():
            if k not in skip and isinstance(v, dict):
                _walk(v, isin, json_file, f"{path}.{k}", records)

    elif ntype == "texte_libre":
        content = _first(node, _CONTENU_KEYS, "") or ""
        records.append({
            "isin": isin,
            "json_file": json_file,
            "json_path": path,
            "node_type": ntype,
            "label_original": "",
            "label": "",
            "value": str(content),
        })

    elif ntype == "sous_section":
        for i, child in enumerate(children):
            _walk(child, isin, json_file, f"{path}.{children_key}[{i}]", records)

    elif ntype == "tableau":
        label = _first(node, _TITRE_KEYS, "")
        if label:
            records.append({
                "isin": isin,
                "json_file": json_file,
                "json_path": path,
                "node_type": ntype,
                "label_original": label,
                "label": _normalize(label),
                "value": str(node.get("_lignes", node.get("_rows", ""))),
            })

    else:
        for k, v in node.items():
            if isinstance(v, list):
                for i, item in enumerate(v):
                    _walk(item, isin, json_file, f"{path}.{k}[{i}]", records)
            elif isinstance(v, dict):
                _walk(v, isin, json_file, f"{path}.{k}", records)


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class SGSearch:
    """
    Outil de recherche dans les JSON SG avec liaison au parquet.

    Parameters
    ----------
    json_dir : str | Path
        Dossier contenant les fichiers .json extraits.
    parquet_path : str | Path
        Chemin vers data_SG.parquet.
    """

    INDEX_FILE = "sg_search_index.parquet"

    def __init__(self, json_dir: str, parquet_path: str):
        self.json_dir = Path(json_dir)
        self.parquet_path = Path(parquet_path)
        self.index: Optional[pd.DataFrame] = None
        self._parquet: Optional[pd.DataFrame] = None

    # ── Index ────────────────────────────────────────────────────────────────

    def build_index(
        self,
        force: bool = False,
        cache_path: Optional[str] = None,
    ) -> "SGSearch":
        """
        Construit (ou recharge depuis cache) l'index plat.

        Parameters
        ----------
        force : bool
            Reconstruire même si le cache est à jour.
        cache_path : str, optional
            Emplacement du fichier cache parquet.
        """
        cache = Path(cache_path) if cache_path else self.json_dir.parent / self.INDEX_FILE
        json_files = sorted(self.json_dir.glob("*.json"))

        # Si aucun JSON mais cache disponible : charger le cache
        if not json_files:
            if cache.exists() and not force:
                self.index = pd.read_parquet(cache)
                print(
                    f"⚠ Aucun JSON dans {self.json_dir} — cache chargé "
                    f"({len(self.index):,} lignes, {self.index['isin'].nunique()} docs)"
                )
                return self
            raise FileNotFoundError(f"Aucun JSON trouvé dans {self.json_dir}")

        # Recharger le cache si plus récent que tous les JSON
        if not force and cache.exists():
            cache_mtime = cache.stat().st_mtime
            if all(f.stat().st_mtime <= cache_mtime for f in json_files):
                self.index = pd.read_parquet(cache)
                print(
                    f"✓ Index chargé depuis cache "
                    f"({len(self.index):,} lignes, {self.index['isin'].nunique()} docs)"
                )
                return self

        # Construction depuis les JSON
        records: list = []
        skipped: list = []
        for f in json_files:
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (json.JSONDecodeError, OSError) as e:
                skipped.append((f.name, str(e).split("\n")[0]))
                continue
            _walk(doc, isin="", json_file=f.name, path="root", records=records)

        cols = ["isin", "json_file", "json_path", "node_type",
                "label_original", "label", "value"]
        self.index = pd.DataFrame(records, columns=cols)
        self.index.to_parquet(cache, index=False)
        n_ok = len(json_files) - len(skipped)
        n_docs = self.index["isin"].nunique() if not self.index.empty else 0
        print(
            f"✓ Index construit "
            f"({len(self.index):,} lignes, {n_docs} docs sur {n_ok}/{len(json_files)} JSON valides)"
            f"\n  Sauvegardé → {cache}"
        )
        if skipped:
            print(f"\n⚠ {len(skipped)} fichier(s) ignoré(s) (JSON invalide) :")
            for name, err in skipped[:10]:
                print(f"  - {name}: {err}")
            if len(skipped) > 10:
                print(f"  ... et {len(skipped) - 10} autre(s)")
            self.skipped_files = skipped
        return self

    def _ensure_index(self) -> None:
        if self.index is None:
            raise RuntimeError("Appelle d'abord build_index()")

    # ── Recherche labels ─────────────────────────────────────────────────────

    def search_labels(
        self,
        pattern: str,
        search_in: str = "label",
        case: bool = False,
        node_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Recherche par regex et retourne les labels candidats.

        Parameters
        ----------
        pattern : str
            Expression régulière (Python re).
        search_in : "label" | "value" | "both"
            - "label"  : recherche dans _cle / _titre (normalisé sans accents)
            - "value"  : recherche dans _valeur / _contenu
            - "both"   : les deux
        case : bool
            Sensible à la casse (défaut False).
        node_types : list[str], optional
            Restreindre aux types : "champ", "avertissement", "section", "texte_libre".

        Returns
        -------
        DataFrame : label_original | node_type | doc_count | value_sample
        """
        self._ensure_index()
        idx = self.index.copy()

        if node_types:
            idx = idx[idx["node_type"].isin(node_types)]

        flags = 0 if case else re.IGNORECASE

        # Pour la recherche sur "label", on travaille sur la version normalisée
        # (pas d'accents) → le pattern n'a pas besoin d'accents non plus
        if search_in == "label":
            norm_pattern = _normalize(pattern)
            mask = idx["label"].str.contains(norm_pattern, flags=flags, regex=True, na=False)
        elif search_in == "value":
            mask = idx["value"].str.contains(pattern, flags=flags, regex=True, na=False)
        else:  # both
            norm_pattern = _normalize(pattern)
            mask = (
                idx["label"].str.contains(norm_pattern, flags=flags, regex=True, na=False)
                | idx["value"].str.contains(pattern, flags=flags, regex=True, na=False)
            )

        matched = idx[mask]

        if matched.empty:
            print("Aucun résultat pour ce pattern.")
            return pd.DataFrame(columns=["label_original", "node_type", "doc_count", "value_sample"])

        agg = (
            matched.groupby(["label_original", "node_type"], sort=False)
            .agg(
                doc_count=("isin", "nunique"),
                value_sample=("value", lambda x: str(x.iloc[0])[:120] if len(x) else ""),
            )
            .reset_index()
            .sort_values("doc_count", ascending=False)
            .reset_index(drop=True)
        )
        return agg

    # ── Valeurs distinctes pour des labels ───────────────────────────────────

    def list_values(
        self,
        labels_df: pd.DataFrame,
        selected_indices: Optional[List[int]] = None,
        selected_labels: Optional[List[str]] = None,
        max_len: int = 200,
    ) -> pd.DataFrame:
        """
        Retourne les valeurs distinctes associées aux labels sélectionnés.

        Permet, après avoir choisi des labels via search_labels(), de visualiser
        toutes les valeurs possibles pour ne conserver que celles qui intéressent
        (ex : "Applicable" vs "Not Applicable").

        Parameters
        ----------
        labels_df : DataFrame
            Résultat de search_labels().
        selected_indices : list[int], optional
            Indices à conserver. Si None et selected_labels None → tout garder.
        selected_labels : list[str], optional
            Alternative : noms exacts des labels à conserver.
        max_len : int
            Tronque l'affichage des valeurs longues (défaut 200).

        Returns
        -------
        DataFrame : value | doc_count | label_count | label_sample
        """
        self._ensure_index()

        if selected_indices is not None:
            kept = labels_df.loc[selected_indices, "label_original"].tolist()
        elif selected_labels is not None:
            kept = selected_labels
        else:
            kept = labels_df["label_original"].tolist()

        matched = self.index[self.index["label_original"].isin(kept)]

        if matched.empty:
            print("Aucune valeur trouvée pour ces labels.")
            return pd.DataFrame(
                columns=["value", "doc_count", "label_count", "label_sample"]
            )

        # Tronque les valeurs très longues (texte_libre, contenus de section)
        # pour l'agrégation lisible, mais on garde la valeur originale comme clé.
        df = matched.copy()
        df["value_display"] = df["value"].astype(str).str.slice(0, max_len)

        agg = (
            df.groupby("value_display", dropna=False, sort=False)
            .agg(
                doc_count=("isin", "nunique"),
                label_count=("label_original", "nunique"),
                label_sample=(
                    "label_original",
                    lambda x: str(x.iloc[0])[:80] if len(x) else "",
                ),
            )
            .reset_index()
            .rename(columns={"value_display": "value"})
            .sort_values("doc_count", ascending=False)
            .reset_index(drop=True)
        )
        return agg

    # ── Filtre ISINs ─────────────────────────────────────────────────────────

    def filter(
        self,
        labels_df: pd.DataFrame,
        selected_indices: Optional[List[int]] = None,
        selected_labels: Optional[List[str]] = None,
        values_df: Optional[pd.DataFrame] = None,
        selected_value_indices: Optional[List[int]] = None,
        selected_values: Optional[List[str]] = None,
        value_regex: Optional[str] = None,
        case: bool = False,
    ) -> List[str]:
        """
        Retourne les ISINs des documents correspondant aux labels sélectionnés.

        Parameters
        ----------
        labels_df : DataFrame
            Résultat de search_labels().
        selected_indices : list[int], optional
            Indices (ligne 0, 1, 2…) à conserver. Si None → tout garder.
        selected_labels : list[str], optional
            Alternative à selected_indices : noms exacts des labels à conserver.
        values_df : DataFrame, optional
            Résultat de list_values() — utilisé conjointement à
            selected_value_indices pour ne garder que certaines valeurs.
        selected_value_indices : list[int], optional
            Indices (du tableau values_df) des valeurs à conserver.
        selected_values : list[str], optional
            Alternative : valeurs exactes (ou préfixes tronqués à max_len) à garder.
        value_regex : str, optional
            Regex supplémentaire filtrée sur la valeur/contenu du nœud.

        Returns
        -------
        list[str] : ISINs correspondants.
        """
        self._ensure_index()

        if selected_indices is not None:
            kept = labels_df.loc[selected_indices, "label_original"].tolist()
        elif selected_labels is not None:
            kept = selected_labels
        else:
            kept = labels_df["label_original"].tolist()

        mask = self.index["label_original"].isin(kept)
        matched = self.index[mask]

        # Filtre par valeurs sélectionnées (depuis list_values)
        kept_values: Optional[List[str]] = None
        if selected_value_indices is not None and values_df is not None:
            kept_values = values_df.loc[selected_value_indices, "value"].tolist()
        elif selected_values is not None:
            kept_values = selected_values

        if kept_values is not None:
            # Comparaison sur la version tronquée pour rester cohérent avec list_values
            # (les valeurs très longues y sont tronquées à max_len caractères).
            kept_set = {str(v) for v in kept_values}
            max_len = max((len(v) for v in kept_set), default=0)
            value_trunc = matched["value"].astype(str).str.slice(0, max_len) if max_len else matched["value"].astype(str)
            matched = matched[
                matched["value"].astype(str).isin(kept_set)
                | value_trunc.isin(kept_set)
            ]

        if value_regex:
            flags = 0 if case else re.IGNORECASE
            matched = matched[
                matched["value"].str.contains(value_regex, flags=flags, regex=True, na=False)
            ]

        isins = sorted(matched["isin"].dropna().unique().tolist())
        total_docs = self.index["isin"].nunique()
        print(f"→ {len(isins)} document(s) sur {total_docs} ({len(isins)/total_docs*100:.1f}%)")
        return isins

    # ── Output parquet ───────────────────────────────────────────────────────

    def to_dataframe(self, isins: List[str]) -> pd.DataFrame:
        """
        Retourne les lignes du parquet correspondant aux ISINs donnés.

        Parameters
        ----------
        isins : list[str]
            Liste retournée par filter().

        Returns
        -------
        DataFrame filtré.
        """
        if self._parquet is None:
            self._parquet = pd.read_parquet(self.parquet_path)

        result = self._parquet[self._parquet["isin"].isin(isins)].copy()
        total = len(self._parquet)
        pct = len(result) / total * 100 if total else 0
        print(f"→ {len(result):,} lignes retenues sur {total:,} ({pct:.1f}%)")
        return result

    # ── Sauvegarde / rechargement filtre ─────────────────────────────────────

    def save_filter(
        self,
        path: str,
        labels: List[str],
        value_regex: Optional[str] = None,
    ) -> None:
        """
        Sauvegarde la configuration du filtre (labels + value_regex) dans un JSON.
        Permet de rejouer la même recherche sur de nouveaux documents.
        """
        config = {"labels": labels, "value_regex": value_regex}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"Filtre sauvegardé → {path}")

    def load_and_apply_filter(self, path: str) -> List[str]:
        """
        Charge un filtre sauvegardé et retourne directement les ISINs correspondants.
        """
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
        return self.filter(
            labels_df=pd.DataFrame(),
            selected_labels=config["labels"],
            value_regex=config.get("value_regex"),
        )
