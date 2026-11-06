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


# --- Niveau 2 : modèle d'images ---------------------------------------------
from pathlib import Path
from PIL import Image

from src.pokedex_data import IMAGE_MODEL, load_image_models, prepare_image, predict_image


def test_pretraitement_image(tmp_path):
    # PNG transparent : le fond doit devenir blanc, l'image carrée 224 x 224
    img = Image.new("RGBA", (300, 120), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (100, 30, 200, 90))
    f = tmp_path / "test.png"
    img.save(f)
    x = prepare_image(f)
    assert x.shape == (224, 224, 3)
    assert x[0, 0].tolist() == [255.0, 255.0, 255.0]


@pytest.mark.skipif(not IMAGE_MODEL.exists(), reason="modèle d'images non disponible")
def test_prediction_image():
    mods = load_image_models()
    # Image synthétique : la prédiction doit être bien formée (probabilités, types connus)
    img = Image.new("RGB", (224, 224), (255, 255, 255))
    img.paste((240, 200, 40), (60, 60, 160, 170))
    r = predict_image(mods, img)
    assert len(r["top"]) == 5 and abs(sum(t["proba"] for t in r["top"])) <= 1 + 1e-6
    assert all(t in mods.type_names for t in r["types_espece"])
    assert 1 <= len(r["types_modele_A"]) <= 2
