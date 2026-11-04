"""Style graphique commun (notebook, rapport, appli).

Règles suivies :
- couleur par rôle : catégoriel (identité) = teintes dans un ordre fixe,
  séquentiel (quantité) = une seule teinte du clair au foncé,
  divergent (signe) = bleu <-> rouge avec un gris neutre au milieu ;
- non-légendaire = bleu, légendaire = orange, partout dans le projet
  (paire vérifiée lisible pour les daltoniens) ;
- traits fins, grille discrète, pas de double axe.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIGURES_DIR

# Palette catégorielle (ordre fixe, jamais recyclé)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Couleurs « métier » du projet
COLOR_NON_LEG = SERIES[0]   # bleu
COLOR_LEG = SERIES[1]       # orange
LEG_PALETTE = {0: COLOR_NON_LEG, 1: COLOR_LEG}
LEG_LABELS = {0: "Non légendaire", 1: "Légendaire"}

# Encres et surfaces
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Rampe séquentielle (bleu) et palette divergente (bleu <-> gris <-> rouge)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_bleu", ["#f4f8fd"] + BLUE_RAMP)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_bleu_rouge", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                       "#f2a9a8", "#e34948", "#a32b2b"])


def set_style() -> None:
    """Applique le style à toutes les figures matplotlib / seaborn."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": INK_2,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "patch.linewidth": 0,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """Enregistre la figure dans reports/figures/<name>.png (pour le rapport)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
