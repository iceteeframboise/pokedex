"""Tests rapides des prédictions utilisées par l'appli (lancer : python -m pytest)."""
import pytest

from src.config import STATS
from src.pokedex_data import load_pokemon, load_tabular_models, nearest_pokemon, predict_stats

MODELS = load_tabular_models()
DF = load_pokemon()


def stats_of(name):
    return DF.loc[DF["name"] == name, STATS].iloc[0].to_dict()


def test_sortie_complete():
    r = predict_stats(MODELS, stats_of("Pikachu"))
    assert 0 <= r["p_legendary"] <= 1
    assert len(r["top3"]) == 3
    assert abs(sum(t["proba"] for t in r["types"]) - 1) < 1e-6
    assert r["cluster"]["name"]


def test_cas_connus():
    assert predict_stats(MODELS, stats_of("Mewtwo"))["legendary_probable"]
    assert not predict_stats(MODELS, stats_of("Magikarp"))["legendary_probable"]
    assert predict_stats(MODELS, stats_of("Shuckle"))["cluster"]["key"] == "tanks"


def test_bornes():
    with pytest.raises(ValueError):
        predict_stats(MODELS, {s: 300 for s in STATS})
    with pytest.raises(ValueError):
        predict_stats(MODELS, {"hp": 50})


def test_plus_proche_est_soi_meme():
    assert nearest_pokemon(DF, MODELS, stats_of("Steelix"), k=1)["name"].item() == "Steelix"
