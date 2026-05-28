"""
app.py — Application Streamlit de recherche de documents SG.

Workflow en 4 étapes :
    1. Recherche de labels par expression régulière
    2. Sélection manuelle des labels à conserver
    3. Sélection manuelle des valeurs associées à conserver
    4. Filtrage final + export Excel (.xlsx)

Lancement :
    streamlit run app.py
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from sg_search import SGSearch


# ─────────────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Recherche SG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Compacter un peu les marges */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* Titres d'étape */
    .step-title {
        display: flex; align-items: center; gap: 0.6rem;
        font-size: 1.35rem; font-weight: 600;
        margin: 0.5rem 0 0.75rem 0;
    }
    .step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 1.9rem; height: 1.9rem; border-radius: 50%;
        background: #1f4e79; color: white;
        font-size: 0.95rem; font-weight: 700;
    }
    .step-num.disabled { background: #c9c9c9; }
    .step-num.done    { background: #2e7d32; }

    /* Boutons primaires plus francs */
    .stButton button[kind="primary"] {
        background: #1f4e79; border-color: #1f4e79;
    }

    /* Tableaux : un peu plus compacts */
    [data-testid="stDataFrame"] { font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Construction de l'index…")
def get_search_engine(json_dir: str, parquet_path: str, _nonce: int = 0) -> SGSearch:
    """Instancie SGSearch et construit/charge l'index. Cache par paramètres."""
    s = SGSearch(json_dir=json_dir, parquet_path=parquet_path)
    # On capture stdout pour ne pas polluer la console (mais on l'expose à la sidebar)
    buf = io.StringIO()
    with redirect_stdout(buf):
        s.build_index()
    s._build_log = buf.getvalue()  # type: ignore[attr-defined]
    return s


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "resultats") -> bytes:
    """Convertit un DataFrame en .xlsx (bytes), prêt pour st.download_button."""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        # Auto-fit grossier
        for i, col in enumerate(df.columns):
            try:
                width = min(60, max(12, int(df[col].astype(str).str.len().quantile(0.95))))
            except Exception:
                width = 18
            ws.set_column(i, i, width)
    return out.getvalue()


def step_header(num: int, title: str, status: str = "active") -> None:
    """Affiche un titre d'étape avec une pastille numérotée."""
    cls = {"done": "step-num done", "disabled": "step-num disabled"}.get(status, "step-num")
    st.markdown(
        f'<div class="step-title"><span class="{cls}">{num}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def reset_downstream(from_step: int) -> None:
    """Vide l'état des étapes ≥ from_step pour rester cohérent."""
    keys_by_step = {
        1: ["labels_df", "values_df", "isins", "result_df"],
        2: ["values_df", "isins", "result_df"],
        3: ["isins", "result_df"],
    }
    for key in keys_by_step.get(from_step, []):
        st.session_state.pop(key, None)
    # Force la réinitialisation des data_editors
    st.session_state["editor_nonce"] = st.session_state.get("editor_nonce", 0) + 1


def selection_summary(df: pd.DataFrame, label: str) -> None:
    """Affiche un petit résumé du nombre d'éléments cochés."""
    n_total = len(df)
    n_kept = int(df["garder"].sum()) if "garder" in df.columns else 0
    color = "#2e7d32" if n_kept else "#b00020"
    st.markdown(
        f"<div style='font-size:0.95rem;color:{color};margin-top:0.25rem;'>"
        f"<b>{n_kept}</b> {label} sélectionné·e·s sur {n_total}.</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar : configuration & index
# ─────────────────────────────────────────────────────────────────────────────

st.session_state.setdefault("editor_nonce", 0)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    json_dir = st.text_input(
        "Dossier des JSON",
        value=st.session_state.get("json_dir", "data/json_SG"),
        help="Dossier contenant les fichiers .json extraits.",
    )
    parquet_path = st.text_input(
        "Fichier parquet",
        value=st.session_state.get("parquet_path", "data/data_SG.parquet"),
        help="Parquet maître contenant tous les documents (data_SG.parquet).",
    )
    st.session_state["json_dir"] = json_dir
    st.session_state["parquet_path"] = parquet_path

    rebuild = st.button(
        "🔄 Reconstruire l'index",
        use_container_width=True,
        help="Force la reconstruction de l'index depuis les JSON.",
    )
    if rebuild:
        get_search_engine.clear()
        for k in ("labels_df", "values_df", "isins", "result_df"):
            st.session_state.pop(k, None)
        st.rerun()

    # Vérifications de chemin
    if not Path(json_dir).exists():
        st.warning(f"Le dossier **{json_dir}** n'existe pas.")
    if not Path(parquet_path).exists():
        st.warning(f"Le parquet **{parquet_path}** n'existe pas.")

    st.divider()

    # Construction / chargement de l'index
    try:
        engine = get_search_engine(json_dir, parquet_path)
    except Exception as exc:
        st.error(f"Impossible de construire l'index : {exc}")
        st.stop()

    if engine.index is not None and not engine.index.empty:
        n_lines = len(engine.index)
        n_docs = engine.index["isin"].nunique()
        c1, c2 = st.columns(2)
        c1.metric("Lignes", f"{n_lines:,}")
        c2.metric("Documents", f"{n_docs}")

    if getattr(engine, "_build_log", ""):
        with st.expander("Log de construction"):
            st.code(engine._build_log, language="text")

    st.divider()
    if st.button("♻️ Réinitialiser la recherche", use_container_width=True):
        for k in ("labels_df", "values_df", "isins", "result_df"):
            st.session_state.pop(k, None)
        st.session_state["editor_nonce"] += 1
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# En-tête principal
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔍 Recherche de documents SG")
st.caption(
    "Filtre les documents par labels et valeurs, puis exporte la sélection au format Excel."
)


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Regex sur les labels
# ─────────────────────────────────────────────────────────────────────────────

step_header(1, "Recherche par expression régulière")

with st.container(border=True):
    with st.form("search_form", border=False, clear_on_submit=False):
        col_regex, col_scope = st.columns([5, 1.3])
        with col_regex:
            regex_input = st.text_input(
                "Expression régulière",
                value=st.session_state.get("regex_input", ""),
                placeholder=r"(?i)Prohibition.*Sales|interdiction.*ventes",
                key="regex_input",
                label_visibility="collapsed",
                help="Regex Python — appuie sur Entrée ou clique sur Rechercher.",
            )
        with col_scope:
            search_in = st.selectbox(
                "Cherche dans",
                options=["label", "value", "both"],
                index=0,
                label_visibility="collapsed",
                help="label = clé/titre · value = valeur/contenu · both = les deux",
            )
        do_search = st.form_submit_button(
            "🔎 Rechercher",
            type="primary",
            use_container_width=True,
        )

    if do_search:
        if not regex_input.strip():
            st.warning("Saisis une expression régulière.")
        else:
            try:
                re.compile(regex_input)
            except re.error as exc:
                st.error(f"Regex invalide : {exc}")
            else:
                with st.spinner("Recherche en cours…"):
                    df_labels = engine.search_labels(regex_input, search_in=search_in)
                if df_labels.empty:
                    st.info("Aucun label ne correspond à cette expression.")
                    reset_downstream(1)
                else:
                    df_labels = df_labels.copy()
                    df_labels.insert(0, "garder", True)
                    st.session_state["labels_df"] = df_labels
                    reset_downstream(2)

    if "labels_df" in st.session_state:
        labels_df: pd.DataFrame = st.session_state["labels_df"]

        st.success(f"**{len(labels_df)}** label(s) trouvé(s). Coche les labels à conserver.")

        b1, b2, _ = st.columns([2, 2, 6])
        if b1.button("✅ Tout cocher", use_container_width=True, key="labels_check_all"):
            labels_df["garder"] = True
            st.session_state["labels_df"] = labels_df
            st.session_state["editor_nonce"] += 1
            reset_downstream(2)
            st.rerun()
        if b2.button("❌ Tout décocher", use_container_width=True, key="labels_uncheck_all"):
            labels_df["garder"] = False
            st.session_state["labels_df"] = labels_df
            st.session_state["editor_nonce"] += 1
            reset_downstream(2)
            st.rerun()

        edited = st.data_editor(
            labels_df,
            column_config={
                "garder": st.column_config.CheckboxColumn(
                    "Garder", default=True, width="small"
                ),
                "label_original": st.column_config.TextColumn("Label", width="large"),
                "node_type": st.column_config.TextColumn("Type", width="small"),
                "doc_count": st.column_config.NumberColumn("Docs", format="%d", width="small"),
                "value_sample": st.column_config.TextColumn("Exemple de valeur"),
            },
            disabled=["label_original", "node_type", "doc_count", "value_sample"],
            use_container_width=True,
            hide_index=False,
            key=f"labels_editor_{st.session_state['editor_nonce']}",
        )
        st.session_state["labels_df"] = edited
        selection_summary(edited, "label(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Valeurs associées aux labels retenus
# ─────────────────────────────────────────────────────────────────────────────

labels_ready = "labels_df" in st.session_state and st.session_state["labels_df"]["garder"].any()
step_header(2, "Sélection des valeurs associées", "active" if labels_ready else "disabled")

with st.container(border=True):
    if not labels_ready:
        st.info("Effectue d'abord une recherche à l'étape 1 et coche au moins un label.")
    else:
        labels_df = st.session_state["labels_df"]
        kept_labels = labels_df[labels_df["garder"]]

        col_a, col_b = st.columns([2, 4])
        with col_a:
            list_values_clicked = st.button(
                "📋 Lister les valeurs",
                type="primary",
                use_container_width=True,
            )
        with col_b:
            st.caption(
                f"{len(kept_labels)} label(s) coché(s) → on en extrait toutes les valeurs distinctes."
            )

        if list_values_clicked:
            with st.spinner("Extraction des valeurs…"):
                values_df = engine.list_values(
                    labels_df, selected_indices=kept_labels.index.tolist()
                )
            if values_df.empty:
                st.warning("Aucune valeur trouvée pour les labels sélectionnés.")
                reset_downstream(2)
            else:
                values_df = values_df.copy()
                values_df.insert(0, "garder", True)
                st.session_state["values_df"] = values_df
                reset_downstream(3)

        if "values_df" in st.session_state:
            values_df = st.session_state["values_df"]

            st.success(
                f"**{len(values_df)}** valeur(s) distincte(s). "
                "Coche celles à conserver."
            )

            b1, b2, _ = st.columns([2, 2, 6])
            if b1.button("✅ Tout cocher", use_container_width=True, key="values_check_all"):
                values_df["garder"] = True
                st.session_state["values_df"] = values_df
                st.session_state["editor_nonce"] += 1
                reset_downstream(3)
                st.rerun()
            if b2.button("❌ Tout décocher", use_container_width=True, key="values_uncheck_all"):
                values_df["garder"] = False
                st.session_state["values_df"] = values_df
                st.session_state["editor_nonce"] += 1
                reset_downstream(3)
                st.rerun()

            edited_v = st.data_editor(
                values_df,
                column_config={
                    "garder": st.column_config.CheckboxColumn(
                        "Garder", default=True, width="small"
                    ),
                    "value": st.column_config.TextColumn("Valeur", width="large"),
                    "doc_count": st.column_config.NumberColumn(
                        "Docs", format="%d", width="small"
                    ),
                    "label_count": st.column_config.NumberColumn(
                        "Labels", format="%d", width="small"
                    ),
                    "label_sample": st.column_config.TextColumn("Exemple de label"),
                },
                disabled=["value", "doc_count", "label_count", "label_sample"],
                use_container_width=True,
                hide_index=False,
                key=f"values_editor_{st.session_state['editor_nonce']}",
            )
            st.session_state["values_df"] = edited_v
            selection_summary(edited_v, "valeur(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 3 — Filtrage final + export
# ─────────────────────────────────────────────────────────────────────────────

values_ready = "values_df" in st.session_state and st.session_state["values_df"]["garder"].any()
step_header(3, "Filtrage final & export Excel", "active" if values_ready else "disabled")

with st.container(border=True):
    if not values_ready:
        st.info("Sélectionne d'abord au moins une valeur à l'étape 2.")
    else:
        labels_df = st.session_state["labels_df"]
        values_df = st.session_state["values_df"]
        kept_labels = labels_df[labels_df["garder"]]
        kept_values = values_df[values_df["garder"]]

        with st.form("filter_form", border=False, clear_on_submit=False):
            value_regex = st.text_input(
                "Regex optionnelle sur la valeur (ex. `Royaume-Uni|UK`)",
                value=st.session_state.get("value_regex", ""),
                key="value_regex",
                placeholder="Laisser vide si non utilisée",
            )
            do_filter = st.form_submit_button(
                "⚡ Filtrer",
                type="primary",
                use_container_width=True,
            )

        st.caption(
            f"{len(kept_labels)} label(s) × {len(kept_values)} valeur(s) → "
            "on récupère les ISINs correspondants puis les lignes du parquet."
        )

        if do_filter:
            with st.spinner("Filtrage des documents…"):
                isins: List[str] = engine.filter(
                    labels_df,
                    selected_indices=kept_labels.index.tolist(),
                    values_df=values_df,
                    selected_value_indices=kept_values.index.tolist(),
                    value_regex=value_regex.strip() or None,
                )
                result_df = engine.to_dataframe(isins)
            st.session_state["isins"] = isins
            st.session_state["result_df"] = result_df

        if "result_df" in st.session_state:
            result_df: pd.DataFrame = st.session_state["result_df"]
            isins = st.session_state["isins"]

            m1, m2, m3 = st.columns(3)
            m1.metric("ISINs uniques", f"{len(isins):,}")
            m2.metric("Lignes parquet", f"{len(result_df):,}")
            total_lines = len(engine._parquet) if engine._parquet is not None else None
            if total_lines:
                pct = len(result_df) / total_lines * 100
                m3.metric("Part du parquet", f"{pct:.1f} %")

            if result_df.empty:
                st.warning("Aucun document ne correspond à cette combinaison.")
            else:
                preview_cols = [
                    c
                    for c in ("isin", "issuer", "document", "fileName", "lastModificationDate")
                    if c in result_df.columns
                ]
                with st.expander("Aperçu (20 premières lignes)", expanded=True):
                    st.dataframe(
                        result_df[preview_cols].head(20) if preview_cols else result_df.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                xlsx_bytes = to_excel_bytes(result_df)
                fname = f"recherche_sg_{datetime.now():%Y%m%d_%H%M}.xlsx"
                st.download_button(
                    "📥 Télécharger en .xlsx",
                    data=xlsx_bytes,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True,
                )
