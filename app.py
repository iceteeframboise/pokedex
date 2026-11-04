"""Mon Pokédex Pokémon : application Streamlit.

Lancement (depuis la racine du projet) :
    streamlit run app.py

L'appli charge les modèles sauvegardés dans models/ (produits par
`python -m src.train_tabular`) et ne réentraîne rien.

Onglets :
    Statistiques  (niveau 1) : 6 stats -> groupe, probabilité d'être légendaire, top-3 des types
    Image         (niveau 2) : à venir
    Caméra        (niveau 3) : à venir
    Pokédex       (niveau 3) : à venir
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import STAT_LABELS, STATS, TYPE_FR
from src.pokedex_data import (STAT_MAX, STAT_MIN, load_pokemon, load_tabular_models,
                              missing_model_files, nearest_pokemon, predict_stats)

st.set_page_config(page_title="Mon Pokédex", page_icon="🔴", layout="wide")

COLOR_POKEMON = "#2a78d6"
COLOR_LEG = "#eb6834"
MUTED = "#898781"
LIBRE = "(saisie libre)"
DEFAULT_POKEMON = "Pikachu"


# --------------------------------------------------------------------------
# Chargement (mis en cache : une seule fois par session du serveur)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des modèles…")
def get_models():
    return load_tabular_models()


@st.cache_data(show_spinner=False)
def get_pokemon() -> pd.DataFrame:
    return load_pokemon()


def label_pokemon(row) -> str:
    return f"#{int(row.pokedex_number):03d} {row['name']}"


# --------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------
def radar(stats: dict, cluster: dict, moyenne: pd.Series, titre: str) -> go.Figure:
    labels = [STAT_LABELS[s] for s in STATS]
    close = lambda v: list(v) + [v[0]]
    v_poke = [stats[s] for s in STATS]
    v_clu = [cluster["profil_moyen"][s] for s in STATS]
    v_all = [moyenne[s] for s in STATS]
    r_max = max(150, max(v_poke + v_clu) * 1.1)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=close(v_all), theta=close(labels), name="Moyenne des 801 Pokémon",
                                  line=dict(color=MUTED, width=1.5, dash="dot"),
                                  hovertemplate="%{theta} : %{r:.0f}<extra>moyenne générale</extra>"))
    fig.add_trace(go.Scatterpolar(r=close(v_clu), theta=close(labels), name=f"Profil moyen : {cluster['name']}",
                                  line=dict(color=cluster["color"], width=2, dash="dash"),
                                  fill="toself", fillcolor=_rgba(cluster["color"], 0.10),
                                  hovertemplate="%{theta} : %{r:.0f}<extra>profil du groupe</extra>"))
    fig.add_trace(go.Scatterpolar(r=close(v_poke), theta=close(labels), name=titre,
                                  line=dict(color=COLOR_POKEMON, width=3), marker=dict(size=7),
                                  fill="toself", fillcolor=_rgba(COLOR_POKEMON, 0.18),
                                  hovertemplate="%{theta} : %{r:.0f}<extra>" + html.escape(titre) + "</extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, r_max], tickfont=dict(size=10), angle=90,
                                                 tickangle=90)),
                      legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
                      margin=dict(l=40, r=40, t=30, b=30), height=430)
    return fig


def bar_types(types: list[dict], vrais: set[str]) -> go.Figure:
    top = types[:3][::-1]
    fig = go.Figure(go.Bar(
        x=[t["proba"] for t in top], y=[t["type_fr"] for t in top], orientation="h",
        marker_color=[COLOR_POKEMON if t["type"] in vrais or not vrais else "#9ec5f4" for t in top],
        text=[f"{t['proba']:.0%}" for t in top], textposition="outside",
        hovertemplate="%{y} : %{x:.1%}<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, max(0.5, top[-1]["proba"] * 1.3)], tickformat=".0%",
                                 title="probabilité"),
                      margin=dict(l=10, r=20, t=10, b=40), height=190, showlegend=False)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------
# Onglet 1 : statistiques
# --------------------------------------------------------------------------
def prefill():
    """Callback : copie les stats du Pokémon choisi dans les champs de saisie."""
    choix = st.session_state["choix_pokemon"]
    if choix != LIBRE:
        df = get_pokemon()
        row = df[df.apply(label_pokemon, axis=1) == choix].iloc[0]
        for s in STATS:
            st.session_state[f"stat_{s}"] = int(row[s])


def tab_statistiques(models, df: pd.DataFrame):
    st.subheader("Analyse à partir des 6 statistiques de base")
    st.caption("Choisissez un Pokémon existant pour préremplir les champs, ou saisissez librement "
               "des statistiques (entre 1 et 255).")

    labels = df.apply(label_pokemon, axis=1).tolist()
    if "choix_pokemon" not in st.session_state:          # première visite : Pikachu
        st.session_state["choix_pokemon"] = labels[df.index[df["name"] == DEFAULT_POKEMON][0]]
        prefill()

    col_in, col_out = st.columns([1, 2], gap="large")

    # ---- Saisie -------------------------------------------------------------
    with col_in:
        st.selectbox("Pokémon existant (facultatif)", [LIBRE] + labels, key="choix_pokemon",
                     on_change=prefill, help="Tapez un nom pour rechercher.")
        for paire in (STATS[0:2], STATS[2:4], STATS[4:6]):     # une ligne par paire : ordre conservé sur mobile
            for col, s in zip(st.columns(2), paire):
                col.number_input(STAT_LABELS[s], min_value=STAT_MIN, max_value=STAT_MAX, step=1, key=f"stat_{s}")
        stats = {s: int(st.session_state[f"stat_{s}"]) for s in STATS}

        choix = st.session_state["choix_pokemon"]
        ref = None
        if choix != LIBRE:
            ref = df[df.apply(label_pokemon, axis=1) == choix].iloc[0]
            if any(int(ref[s]) != stats[s] for s in STATS):
                st.info(f"Stats modifiées par rapport à {ref['name']} : analyse d'un Pokémon « imaginaire ».")
                ref = None
        st.metric("Total des stats", sum(stats.values()),
                  delta=f"{sum(stats.values()) - df['base_total'].median():+.0f} vs médiane",
                  delta_color="off")

    res = predict_stats(models, stats)
    titre = ref["name"] if ref is not None else "Pokémon saisi"

    # ---- Résultats ----------------------------------------------------------
    with col_out:
        g1, g2, g3 = st.columns(3, gap="medium")

        with g1, st.container(border=True):
            cl = res["cluster"]
            st.markdown("**Groupe**")
            st.markdown(f"<div style='border-left:6px solid {cl['color']};padding-left:10px;"
                        f"font-size:1.15rem;font-weight:600'>{html.escape(cl['name'])}</div>",
                        unsafe_allow_html=True)
            st.write(cl["description"])
            st.caption(f"{cl['effectif']} Pokémon dans ce groupe, dont {cl['pct_legendaires']:.0f} % de "
                       f"légendaires. Groupe voisin le plus proche : {res['cluster_second']}.")

        with g2, st.container(border=True):
            p = res["p_legendary"]
            st.markdown("**Probabilité d'être légendaire**")
            st.markdown(f"<div style='font-size:2.2rem;font-weight:700;line-height:1.1'>{p:.0%}</div>",
                        unsafe_allow_html=True)
            st.progress(min(max(p, 0.0), 1.0))
            if res["legendary_probable"]:
                st.markdown(f"<span style='color:{COLOR_LEG};font-weight:600'>⭐ Légendaire probable</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("Probablement **pas** légendaire")
            st.caption(f"Verdict « légendaire probable » au-dessus de {res['threshold']:.0%} "
                       "(seuil qui maximise le F1). Probabilité calibrée : parmi les Pokémon notés 80 %, "
                       "environ 8 sur 10 sont légendaires.")
            if ref is not None:
                st.caption(f"Réalité : {ref['name']} **{'est' if ref['is_legendary'] else 'n’est pas'}** légendaire.")

        with g3, st.container(border=True):
            st.markdown("**Type principal prédit**")
            vrais = {ref["type1"]} if ref is not None else set()
            st.plotly_chart(bar_types(res["types"], vrais), config={"displayModeBar": False},
                            key="bar_types")
            if ref is not None:
                vrai = TYPE_FR.get(ref["type1"], ref["type1"])
                rang = next((i + 1 for i, t in enumerate(res["types"]) if t["type"] == ref["type1"]), None)
                txt = f"Vrai type : **{vrai}**"
                if ref["type2"] != "aucun":
                    txt += f" / {TYPE_FR.get(ref['type2'], ref['type2'])}"
                txt += f" (classé {rang}ᵉ par le modèle)." if rang else " (type non prédit par le modèle)."
                st.caption(txt)
            st.caption("⚠️ Les stats seules prédisent mal le type : le bon type est dans ce top-3 "
                       "environ 1 fois sur 2.")

        r1, r2 = st.columns([3, 2], gap="medium")
        with r1, st.container(border=True):
            st.markdown("**Profil des stats comparé au groupe**")
            st.plotly_chart(radar(stats, res["cluster"], df[STATS].mean(), titre), key="radar")
        with r2, st.container(border=True):
            st.markdown("**Pokémon existants les plus proches**")
            proches = nearest_pokemon(df, models, stats, k=5, exclude_name=ref["name"] if ref is not None else None)
            st.dataframe(proches.assign(
                    Pokémon=proches["name"],
                    Type=proches["type1"].map(TYPE_FR) + np.where(proches["type2"] != "aucun",
                                                                  " / " + proches["type2"].map(TYPE_FR), ""),
                    Total=proches["base_total"],
                    Lég=np.where(proches["is_legendary"] == 1, "oui", "non"),
                    Distance=proches["distance"].round(2))[["Pokémon", "Type", "Total", "Lég", "Distance"]],
                hide_index=True)
            st.caption("Distance euclidienne sur les stats standardisées (la même que celle du K-means).")


def tab_a_venir(niveau: str, texte: str):
    st.subheader(f"{niveau} : à venir")
    st.info(texte)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def main():
    st.title("🔴 Mon Pokédex")
    missing = missing_model_files()
    if missing:
        st.error("Modèles introuvables : " + ", ".join(missing) +
                 ". Lancez d'abord `python -m src.train_tabular` depuis la racine du projet.")
        st.stop()
    models, df = get_models(), get_pokemon()

    t1, t2, t3, t4 = st.tabs(["📊 Statistiques", "🖼️ Image", "📷 Caméra", "📚 Pokédex"])
    with t1:
        tab_statistiques(models, df)
    with t2:
        tab_a_venir("Niveau 2", "Upload d'une image → types prédits avec leurs probabilités.")
    with t3:
        tab_a_venir("Niveau 3", "Capture par la webcam → identification du Pokémon et fiche complète.")
    with t4:
        tab_a_venir("Niveau 3", "Catalogue des 801 Pokémon : filtres, tri, recherche, organisation par types.")

    meta = models.metadata
    st.caption(f"Modèles entraînés le {meta['date_entrainement'][:10]} sur {meta['n_pokemon']} Pokémon "
               f"(scikit-learn {meta['versions']['scikit-learn']}). Aucune donnée n'est réentraînée par l'appli.")


main()
