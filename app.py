"""Mon Pokédex Pokémon : application Streamlit.

Lancement (depuis la racine du projet) :
    streamlit run app.py

L'appli charge les modèles sauvegardés dans models/ (produits par
`python -m src.train_tabular`) et ne réentraîne rien.

Onglets :
    Statistiques  (niveau 1) : 6 stats -> groupe, probabilité d'être légendaire, top-3 des types
    Image         (niveau 2) : image importée -> Pokémon reconnu (top-3) et types prédits
    Caméra        (niveau 3) : photo par la webcam -> Pokémon identifié (ou « non reconnu ») et fiche complète
    Pokédex       (niveau 3) : catalogue des 801 Pokémon (filtres, tri, recherche, fiche) et vue par types
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import STAT_LABELS, STATS, TYPE_FR
from src.pokedex_data import (IMAGE_CLASSES, IMAGE_MODEL, STAT_MAX, STAT_MIN, load_image_models,
                              load_pokemon, load_tabular_models, missing_model_files, nearest_pokemon,
                              pokemon_card, predict_image, predict_stats)

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


@st.cache_resource(show_spinner="Chargement du réseau de neurones (≈ 10 s la première fois)…")
def get_image_models():
    return load_image_models()


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


# --------------------------------------------------------------------------
# Onglet 2 : image
# --------------------------------------------------------------------------
def nom_types(types) -> str:
    return " / ".join(TYPE_FR.get(t, t) for t in types if t != "aucun")


def bar_probas(labels, probas, couleurs=None, height=200) -> go.Figure:
    fig = go.Figure(go.Bar(x=probas[::-1], y=labels[::-1], orientation="h",
                           marker_color=(couleurs or [COLOR_POKEMON] * len(labels))[::-1],
                           text=[f"{p:.0%}" for p in probas[::-1]], textposition="outside",
                           hovertemplate="%{y} : %{x:.1%}<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 1.15], tickformat=".0%"), margin=dict(l=10, r=10, t=10, b=30),
                      height=height, showlegend=False)
    return fig


def afficher_resultat_image(res: dict, models, df: pd.DataFrame, cle: str):
    """Affiche le top-3 des espèces, les types et une fiche rapide (partagé par Image et Caméra)."""
    top = res["top"]
    p1 = top[0]["proba"]
    c1, c2 = st.columns([1, 1], gap="large")
    with c1, st.container(border=True):
        st.markdown("**Pokémon reconnu (top-3)**")
        if p1 >= 0.5:
            st.markdown(f"<div style='font-size:1.6rem;font-weight:700'>#{top[0]['pokedex_number']:03d} "
                        f"{html.escape(top[0]['name'])}</div>", unsafe_allow_html=True)
        else:
            st.warning("Le modèle hésite : la meilleure proposition a moins de 50 % de probabilité.")
        st.plotly_chart(bar_probas([f"#{t['pokedex_number']:03d} {t['name']}" for t in top[:3]],
                                   [t["proba"] for t in top[:3]]), config={"displayModeBar": False},
                        key=f"top3_{cle}")
        st.caption("Modèle : EfficientNetB0 fine-tuné, 142 Pokémon de 1re génération "
                   "(bonne espèce dans 91 % des cas sur le jeu de test, dans le top-3 dans ≈ 96 %).")
    with c2, st.container(border=True):
        st.markdown("**Type(s) prédit(s)**")
        st.markdown(f"<div style='font-size:1.4rem;font-weight:700'>{html.escape(nom_types(res['types_espece']))}</div>",
                    unsafe_allow_html=True)
        st.caption("Types du Pokémon reconnu (lus dans pokemon.csv). Probabilité de chaque type = somme des "
                   "probabilités des Pokémon qui l'ont :")
        pt = list(res["proba_types_espece"].items())[:4]
        st.plotly_chart(bar_probas([TYPE_FR[t] for t, _ in pt], [v for _, v in pt], height=170),
                        config={"displayModeBar": False}, key=f"types_{cle}")
        if "types_modele_A" in res:
            pa = list(res["proba_types_modele_A"].items())[:3]
            accord = set(res["types_modele_A"]) == set(res["types_espece"])
            st.markdown(f"Second avis (modèle qui prédit les types directement) : **{nom_types(res['types_modele_A'])}** "
                        + ("✅ concordant" if accord else "⚠️ différent"))
            st.caption(" · ".join(f"{TYPE_FR[t]} {v:.0%}" for t, v in pa))

    afficher_fiche(top[0]["pokedex_number"], models, df, cle)


# --------------------------------------------------------------------------
# Fiche complète d'un Pokémon (onglets Image, Caméra et Pokédex)
# --------------------------------------------------------------------------
def puces(items: dict | list, fond: str) -> str:
    """Petites étiquettes HTML : {type: multiplicateur} ou [type]."""
    if not items:
        return "<span style='color:#898781'>aucune</span>"
    if isinstance(items, dict):
        items = [f"{TYPE_FR.get(t, t)} ×{m:g}" for t, m in items.items()]
    else:
        items = [TYPE_FR.get(t, t) for t in items]
    return " ".join(f"<span style='background:{fond};border-radius:10px;padding:2px 9px;margin:2px;"
                    f"display:inline-block;font-size:0.85rem'>{html.escape(x)}</span>" for x in items)


def afficher_fiche(pokedex_number: int, models, df: pd.DataFrame, cle: str):
    c = pokemon_card(df, pokedex_number)
    r = predict_stats(models, c["stats"])
    with st.container(border=True):
        st.markdown(f"### #{c['pokedex_number']:03d} {html.escape(c['name'])}")
        st.caption(f"Nom japonais : {c['japanese_kana']} ({c['japanese_romaji']}) · {c['classification']}")
        g1, g2, g3 = st.columns([1, 1.3, 1.7], gap="medium")
        with g1:
            if c["reference_image"]:
                st.image(str(c["reference_image"]), width=180, caption="Image de référence (dataset)")
            st.markdown(f"<div style='font-size:1.15rem;font-weight:700'>{html.escape(nom_types(c['types']))}</div>",
                        unsafe_allow_html=True)
            if c["is_legendary"]:
                st.markdown(f"<span style='color:{COLOR_LEG};font-weight:600'>⭐ Pokémon légendaire</span>",
                            unsafe_allow_html=True)
        with g2:
            sexe = ("asexué" if c["percentage_male"] is None
                    else f"{c['percentage_male']:g} % mâles / {100 - c['percentage_male']:g} % femelles")
            lignes = [("Taille", f"{c['height_m']:g} m" if c["height_m"] is not None else "non renseignée"),
                      ("Poids", f"{c['weight_kg']:g} kg" if c["weight_kg"] is not None else "non renseigné"),
                      ("Talents", ", ".join(c["abilities"])),
                      ("Génération", str(c["generation"])),
                      ("Sexe", sexe),
                      ("Taux de capture", f"{c['capture_rate']} / 255" if c["capture_rate"] is not None else "-"),
                      ("Pas pour éclore", f"{c['base_egg_steps']:,}".replace(",", " ")),
                      ("Bonheur de base", str(c["base_happiness"])),
                      ("Expérience au niveau 100", f"{c['experience_growth']:,}".replace(",", " ")),
                      ("Total des stats", str(c["base_total"]))]
            st.markdown("\n".join(f"- **{k}** : {html.escape(v)}" for k, v in lignes))
            st.markdown(f"**Groupe (niveau 1)** : {r['cluster']['name']}  \n"
                        f"**P(légendaire) prédite par les stats** : {r['p_legendary']:.0%}")
        with g3:
            st.plotly_chart(radar(c["stats"], r["cluster"], df[STATS].mean(), c["name"]),
                            config={"displayModeBar": False}, key=f"radar_{cle}")
        st.markdown("**Faiblesses** (dégâts reçus) : " + puces(c["weaknesses"], "#f6d5d5"), unsafe_allow_html=True)
        st.markdown("**Résistances** : " + puces(c["resistances"], "#d5ecd8"), unsafe_allow_html=True)
        st.markdown("**Immunités** : " + puces(c["immunities"], "#e1e0d9"), unsafe_allow_html=True)
        st.caption("Talents : le jeu de données mélange ceux de la forme classique et de ses formes régionales.")


def tab_image(models, df: pd.DataFrame):
    st.subheader("Reconnaître un Pokémon à partir d'une image")
    st.caption("Importez une image d'un Pokémon de 1re génération (dessin, capture, photo…). Elle est préparée "
               "exactement comme à l'entraînement : fond blanc, carré, 224 × 224 pixels.")
    if not IMAGE_MODEL.exists() or not IMAGE_CLASSES.exists():
        st.error("Modèle d'images introuvable (models/image_model.keras). Voir notebooks/2_images.ipynb, partie 6.")
        return
    fichier = st.file_uploader("Image du Pokémon", type=["png", "jpg", "jpeg", "gif", "webp", "bmp"],
                               key="upload_image")
    if fichier is None:
        st.info("En attente d'une image. Astuce : une image où le Pokémon est bien visible et occupe une "
                "bonne partie du cadre donne les meilleurs résultats.")
        return
    try:
        res = predict_image(get_image_models(), fichier, mode="pad")
    except Exception as e:                                    # fichier illisible, format exotique…
        st.error(f"Impossible de lire cette image : {e}")
        return
    a, b, _ = st.columns([1, 1, 2])
    a.image(fichier, caption="Image importée", width=260)
    b.image(res["image"], caption="Ce que voit le modèle (224 × 224)", width=224)
    afficher_resultat_image(res, models, df, "image")


# --------------------------------------------------------------------------
# Onglet 3 : caméra
# --------------------------------------------------------------------------
def seuil_confiance() -> float:
    import json
    return float(json.loads(IMAGE_CLASSES.read_text(encoding="utf-8")).get("seuil_confiance", 0.7))


def tab_camera(models, df: pd.DataFrame):
    st.subheader("Identifier un Pokémon avec la caméra")
    st.caption("Présentez une image de Pokémon (écran de téléphone, carte, impression…) bien au centre et assez "
               "grande dans le cadre, puis prenez la photo. L'image est recadrée au centre (carré) et préparée comme "
               "à l'entraînement.")
    if not IMAGE_MODEL.exists():
        st.error("Modèle d'images introuvable (models/image_model.keras).")
        return
    tau = seuil_confiance()
    photo = st.camera_input("Caméra", key="camera", label_visibility="collapsed")
    if photo is None:
        st.info(f"Le navigateur va demander l'autorisation d'utiliser la caméra. Une réponse n'est donnée que si le "
                f"modèle est sûr à au moins {tau:.0%} ; sinon : « Pokémon non reconnu ».")
        return
    res = predict_image(get_image_models(), photo, mode="center_crop")
    top = res["top"]
    a, b = st.columns([1, 3])
    a.image(res["image"], caption="Ce que voit le modèle (recadrage centré)", width=224)
    with b:
        if top[0]["proba"] < tau:
            st.error(f"**Pokémon non reconnu** : la meilleure proposition n'atteint que {top[0]['proba']:.0%} de "
                     f"confiance (seuil : {tau:.0%}).")
            st.markdown("Conseils : rapprochez l'image, évitez les reflets et le contre-jour, centrez le Pokémon. "
                        "Le modèle ne connaît que 142 Pokémon de 1re génération.")
            st.markdown("Pistes : " + ", ".join(f"{t['name']} ({t['proba']:.0%})" for t in top[:3]))
            return
        st.success(f"**#{top[0]['pokedex_number']:03d} {top[0]['name']}** reconnu avec {top[0]['proba']:.0%} de confiance")
        st.plotly_chart(bar_probas([f"#{t['pokedex_number']:03d} {t['name']}" for t in top[:3]],
                                   [t["proba"] for t in top[:3]], height=160), config={"displayModeBar": False},
                        key="top3_camera")
    afficher_fiche(top[0]["pokedex_number"], models, df, "camera")


# --------------------------------------------------------------------------
# Onglet 4 : Pokédex (catalogue)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def catalogue(_models, df: pd.DataFrame) -> pd.DataFrame:
    """Table du catalogue : une ligne par Pokémon, avec groupe (niveau 1) et vignette."""
    import base64
    from src.pokedex_data import reference_image
    z = _models.scaler.transform(df[STATS])
    groupes = [_models.clusters[str(c)]["name"] for c in _models.kmeans.predict(z)]

    def vignette(n):
        f = reference_image(n)
        if f is None:                     # pas d'image hors 1re génération : vignette transparente
            return ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
        from PIL import Image
        import io
        b = io.BytesIO()
        Image.open(f).resize((56, 56)).save(b, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()

    cat = pd.DataFrame({
        "image": [vignette(n) for n in df["pokedex_number"]],
        "n°": df["pokedex_number"].astype(int), "nom": df["name"],
        "types": [nom_types([a, b]) for a, b in zip(df["type1"], df["type2"])],
        "type1": df["type1"], "type2": df["type2"],
        "total": df["base_total"].astype(int), **{STAT_LABELS[s_]: df[s_].astype(int) for s_ in STATS},
        "taille (m)": df["height_m"], "poids (kg)": df["weight_kg"],
        "génération": df["generation"].astype(int), "légendaire": df["is_legendary"].astype(bool),
        "groupe": groupes})
    return cat


def tab_pokedex(models, df: pd.DataFrame):
    st.subheader("Pokédex : les 801 Pokémon")
    cat = catalogue(models, df)
    vue = st.segmented_control("Vue", ["Catalogue", "Organisation par types"], default="Catalogue",
                               key="vue_pokedex", label_visibility="collapsed")
    if vue == "Organisation par types":
        vue_types(cat)
        return

    # ---- Filtres -------------------------------------------------------------
    tous_types = sorted(set(cat["type1"]) | (set(cat["type2"]) - {"aucun"}), key=lambda t: TYPE_FR.get(t, t))
    f1, f2, f3, f4 = st.columns([1.3, 1, 0.9, 1.3])
    recherche = f1.text_input("Rechercher un nom", placeholder="ex. pika, dra…", key="recherche")
    types = f2.multiselect("Type (1er ou 2e)", tous_types, format_func=lambda t: TYPE_FR.get(t, t), key="f_types",
                           placeholder="Tous")
    gens = f3.multiselect("Génération", list(range(1, 8)), key="f_gen", placeholder="Toutes")
    groupes = f4.multiselect("Groupe (niveau 1)", sorted(cat["groupe"].unique()), key="f_groupe",
                             placeholder="Tous")
    g1, g2, g3, _ = st.columns([1, 1.3, 1, 1.3])
    leg = g1.radio("Légendaire", ["tous", "oui", "non"], horizontal=True, key="f_leg")
    cles_tri = ["n°", "nom", "total", *[STAT_LABELS[s_] for s_ in STATS], "taille (m)", "poids (kg)"]
    tri = g2.selectbox("Trier par", cles_tri, key="tri")
    ordre = g3.radio("Ordre", ["croissant", "décroissant"], horizontal=True, key="ordre",
                     index=0 if tri in ("n°", "nom") else 1)

    m = pd.Series(True, index=cat.index)
    if recherche:
        m &= cat["nom"].str.contains(recherche.strip(), case=False, regex=False)
    if types:
        m &= cat["type1"].isin(types) | cat["type2"].isin(types)
    if gens:
        m &= cat["génération"].isin(gens)
    if groupes:
        m &= cat["groupe"].isin(groupes)
    if leg != "tous":
        m &= cat["légendaire"] == (leg == "oui")
    res = cat[m].sort_values(tri, ascending=(ordre == "croissant"), na_position="last")

    st.caption(f"**{len(res)}** Pokémon sur {len(cat)}. Cliquez sur une ligne pour afficher sa fiche complète.")
    if res.empty:
        st.info("Aucun Pokémon ne correspond à ces filtres.")
        return
    colonnes = ["image", "n°", "nom", "types", "total", *[STAT_LABELS[s_] for s_ in STATS], "groupe", "légendaire",
                "génération"]
    sel = st.dataframe(
        res[colonnes], hide_index=True, height=420, key="table_pokedex", on_select="rerun",
        selection_mode="single-row",
        column_config={"image": st.column_config.ImageColumn("", width="small"),
                       "n°": st.column_config.NumberColumn(format="%d", width="small"),
                       "total": st.column_config.ProgressColumn("total", min_value=150, max_value=780, format="%d"),
                       "légendaire": st.column_config.CheckboxColumn("lég."),
                       "génération": st.column_config.NumberColumn("gén.", format="%d", width="small")})
    lignes = sel.selection.rows if sel is not None else []
    if lignes:
        n = int(res.iloc[lignes[0]]["n°"])
        afficher_fiche(n, models, df, f"pokedex_{n}")


def vue_types(cat: pd.DataFrame):
    """Organisation par types : effectifs, stats moyennes (heatmap), Pokémon de chaque type."""
    lignes = []
    for t in sorted(set(cat["type1"]) | (set(cat["type2"]) - {"aucun"})):
        m = (cat["type1"] == t) | (cat["type2"] == t)
        lignes.append({"type": TYPE_FR.get(t, t), "code": t, "Pokémon": int(m.sum()),
                       "en type 1": int((cat["type1"] == t).sum()), "en type 2": int((cat["type2"] == t).sum()),
                       "légendaires": int(cat.loc[m, "légendaire"].sum()),
                       **{STAT_LABELS[s_]: cat.loc[m, STAT_LABELS[s_]].mean() for s_ in STATS},
                       "total moyen": cat.loc[m, "total"].mean()})
    par_type = pd.DataFrame(lignes).sort_values("Pokémon", ascending=False)

    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.markdown("**Nombre de Pokémon par type** (un Pokémon à deux types compte pour chacun)")
        fig = go.Figure()
        fig.add_bar(y=par_type["type"], x=par_type["en type 1"], name="type 1", orientation="h",
                    marker_color=COLOR_POKEMON)
        fig.add_bar(y=par_type["type"], x=par_type["en type 2"], name="type 2", orientation="h",
                    marker_color="#9ec5f4")
        fig.update_layout(barmode="stack", height=520, margin=dict(l=10, r=10, t=10, b=30),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, config={"displayModeBar": False}, key="types_effectifs")
    with c2:
        st.markdown("**Stats moyennes par type** (écart à la moyenne des 801 Pokémon, en écarts-types)")
        cols = [STAT_LABELS[s_] for s_ in STATS]
        moy, et = cat[cols].mean(), cat[cols].std()
        z = (par_type.set_index("type")[cols] - moy) / et
        brut = par_type.set_index("type")[cols]
        fig = go.Figure(go.Heatmap(z=z.values, x=cols, y=z.index, zmid=0, zmin=-1.5, zmax=1.5,
                                   colorscale=[[0, "#104281"], [0.5, "#f0efec"], [1, "#a32b2b"]],
                                   text=brut.round(0).astype(int).values, texttemplate="%{text}",
                                   hovertemplate="%{y} · %{x} : %{text} (z = %{z:.2f})<extra></extra>",
                                   colorbar=dict(title="z")))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=30), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, config={"displayModeBar": False}, key="types_heatmap")
        st.caption("Chiffres : moyenne brute de la stat ; couleur : rouge = au-dessus de la moyenne, bleu = en dessous.")

    st.markdown("**Pokémon d'un type**")
    t = st.selectbox("Type", par_type["code"], format_func=lambda c: TYPE_FR.get(c, c), key="type_detail",
                     label_visibility="collapsed")
    sous = cat[(cat["type1"] == t) | (cat["type2"] == t)].sort_values("total", ascending=False)
    st.caption(f"{len(sous)} Pokémon de type {TYPE_FR.get(t, t)}, du plus fort au plus faible (total des stats).")
    st.dataframe(sous[["image", "n°", "nom", "types", "total", "groupe", "légendaire"]], hide_index=True, height=320,
                 column_config={"image": st.column_config.ImageColumn("", width="small"),
                                "total": st.column_config.ProgressColumn("total", min_value=150, max_value=780,
                                                                         format="%d"),
                                "légendaire": st.column_config.CheckboxColumn("lég.")}, key="table_type")


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
        tab_image(models, df)
    with t3:
        tab_camera(models, df)
    with t4:
        tab_pokedex(models, df)

    meta = models.metadata
    st.caption(f"Modèles entraînés le {meta['date_entrainement'][:10]} sur {meta['n_pokemon']} Pokémon "
               f"(scikit-learn {meta['versions']['scikit-learn']}). Aucune donnée n'est réentraînée par l'appli.")


main()
