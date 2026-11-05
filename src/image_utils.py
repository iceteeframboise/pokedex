"""Outils pour les images de Pokémon (niveaux 2 et 3).

Contenu :
- association des dossiers d'images aux lignes de pokemon.csv (noms normalisés) ;
- lecture robuste d'une image (PNG transparent -> fond blanc, GIF -> 1re image,
  CMYK, 16 bits…) ;
- mise au format carré (remplissage blanc pour le dataset, recadrage centré
  pour la caméra) ;
- inventaire des fichiers (corrompus, formats, tailles) et empreintes
  (MD5 pour les doublons exacts, hash perceptuel pour les quasi-doublons) ;
- regroupement des quasi-doublons (union-find) et export du dataset nettoyé.

Le même prétraitement (`load_rgb` puis `to_square`) sert à l'entraînement,
dans l'appli et pour la caméra : c'est indispensable pour que le modèle voie
en production des images préparées exactement comme à l'entraînement.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

from src.config import ROOT_DIR

IMAGES_RAW = ROOT_DIR / "data" / "images" / "pokemon"      # archive Kaggle décompressée
IMAGES_CLEAN = ROOT_DIR / "data" / "images_clean"         # images nettoyées 224 x 224
IMAGES_INDEX = ROOT_DIR / "data" / "images_index.csv"     # inventaire (versionné)
KAGGLE_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
              "mikoajkolman/pokemon-images-first-generation17000-files")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
IMG_SIZE = 224
BACKGROUND = (255, 255, 255)
ImageFile.LOAD_TRUNCATED_IMAGES = False      # une image tronquée doit lever une erreur
Image.MAX_IMAGE_PIXELS = 50_000_000          # garde-fou contre les images démesurées


# ==========================================================================
# 1. Noms : dossiers d'images <-> pokemon.csv
# ==========================================================================
def normalize_name(name: str) -> str:
    """Clé de comparaison : minuscules, sans accents ni ponctuation.

    « Nidoran♀ » -> « nidoranf », « Mr. Mime » et « MrMime » -> « mrmime »,
    « Farfetch'd » et « Farfetchd » -> « farfetchd ».
    """
    name = name.replace("♀", "f").replace("♂", "m")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_folders(images_root: Path, pokemon: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Associe chaque dossier d'images à un Pokémon de la 1re génération.

    Renvoie (association dossier -> Pokémon, dossiers non appariés, Pokémon sans dossier).
    Plusieurs dossiers peuvent désigner le même Pokémon (ex. « Mr. Mime » et « MrMime »).
    """
    gen1 = pokemon[pokemon["pokedex_number"] <= 151]
    par_cle = {normalize_name(n): row for n, (_, row) in zip(gen1["name"], gen1.iterrows())}
    lignes, non_apparies = [], []
    for d in sorted(p.name for p in Path(images_root).iterdir() if p.is_dir()):
        row = par_cle.get(normalize_name(d))
        if row is None:
            non_apparies.append(d)
            continue
        lignes.append({"folder": d, "pokedex_number": int(row["pokedex_number"]), "name": row["name"],
                       "type1": row["type1"], "type2": row["type2"]})
    assoc = pd.DataFrame(lignes)
    manquants = gen1[~gen1["pokedex_number"].isin(assoc["pokedex_number"])][["pokedex_number", "name", "type1", "type2"]]
    return assoc, non_apparies, manquants


# ==========================================================================
# 2. Lecture et mise au format
# ==========================================================================
def load_rgb(source, background=BACKGROUND) -> Image.Image:
    """Ouvre une image (chemin, fichier ou PIL) et renvoie une image RGB sur fond blanc.

    - GIF animé : on garde la 1re image ;
    - transparence (RGBA, LA, P avec transparence) : composition sur fond blanc.
      Sans cela, les pixels transparents deviennent NOIRS lors d'une simple
      conversion en RGB, et le modèle apprendrait « fond noir = PNG » ;
    - orientation EXIF corrigée (photos de téléphone).
    """
    img = source if isinstance(source, Image.Image) else Image.open(source)
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I", "F"):                       # images 16/32 bits
        arr = np.asarray(img, dtype=np.float32)
        arr = 255 * (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0)
        img = Image.fromarray(arr.astype(np.uint8))
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        img = img.convert("RGBA")
        fond = Image.new("RGBA", img.size, background + (255,))
        img = Image.alpha_composite(fond, img)
    return img.convert("RGB")


def to_square(img: Image.Image, size: int = IMG_SIZE, mode: str = "pad",
              background=BACKGROUND) -> Image.Image:
    """Met une image au format carré size x size.

    mode="pad"         : on complète par du blanc (dataset : on ne coupe jamais le Pokémon)
    mode="center_crop" : on garde le carré central (caméra : le Pokémon est au centre du cadre)
    """
    w, h = img.size
    if mode == "center_crop":
        c = min(w, h)
        img = img.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c))
    elif mode == "pad":
        c = max(w, h)
        carre = Image.new("RGB", (c, c), background)
        carre.paste(img, ((c - w) // 2, (c - h) // 2))
        img = carre
    else:
        raise ValueError(mode)
    return img.resize((size, size), Image.LANCZOS)


def has_transparency(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


# ==========================================================================
# 3. Inventaire et empreintes
# ==========================================================================
def _scan_one(path: str) -> dict:
    p = Path(path)
    out = {"path": str(p.relative_to(IMAGES_RAW.parent)), "folder": p.parent.name,
           "ext": p.suffix.lower(), "ok": False, "error": ""}
    if out["ext"] not in IMAGE_EXT:
        out["error"] = "format non image"
        return out
    try:
        raw = p.read_bytes()
        out["md5"] = hashlib.md5(raw).hexdigest()
        with Image.open(p) as im:
            im.verify()                                   # contrôle d'intégrité
        with Image.open(p) as im:
            out.update(format=im.format, mode=im.mode, width=im.width, height=im.height,
                       n_frames=getattr(im, "n_frames", 1), transparency=has_transparency(im))
            rgb = load_rgb(im)                            # décodage complet
        sq = to_square(rgb, 64)
        out["phash"] = str(imagehash.phash(sq))           # 64 bits, en hexadécimal
        arr = np.asarray(sq, dtype=np.float32)
        out["std_pixels"] = float(arr.std())              # ~0 : image vide / unie
        out["ok"] = True
    except Exception as e:                                # fichier corrompu, tronqué, illisible…
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def scan_images(images_root: Path = IMAGES_RAW, n_jobs: int = 4) -> pd.DataFrame:
    """Passe en revue tous les fichiers (lecture complète, empreintes)."""
    paths = sorted(str(p) for p in Path(images_root).rglob("*") if p.is_file())
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        rows = list(ex.map(_scan_one, paths, chunksize=64))
    return pd.DataFrame(rows)


def hamming_matrix_groups(hashes_hex: list[str], max_dist: int, labels=None) -> np.ndarray:
    """Groupes de quasi-doublons : deux images sont reliées si la distance de
    Hamming entre leurs hash perceptuels (64 bits) est <= max_dist (et, si
    `labels` est donné, si elles ont la même étiquette). Les groupes sont les
    composantes connexes (union-find). Renvoie un numéro de groupe par image.
    """
    h = np.array([int(x, 16) for x in hashes_hex], dtype=np.uint64)
    labels = None if labels is None else np.asarray(labels)
    n = len(h)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        d = np.bitwise_count(h[i + 1:] ^ h[i])           # distance de Hamming vectorisée
        proche = d <= max_dist
        if labels is not None:
            proche &= labels[i + 1:] == labels[i]
        for j in np.nonzero(proche)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def min_hamming_to_others(hashes_hex: list[str], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pour chaque image : distance minimale à une image de la MÊME espèce et
    à une image d'une AUTRE espèce (sert à choisir le seuil de quasi-doublon)."""
    h = np.array([int(x, 16) for x in hashes_hex], dtype=np.uint64)
    meme, autre = np.full(len(h), 64), np.full(len(h), 64)
    for i in range(len(h)):
        d = np.bitwise_count(h ^ h[i]).astype(int)
        d[i] = 99
        s = labels == labels[i]
        meme[i] = d[s].min() if s.sum() > 1 else 64
        autre[i] = d[~s].min()
    return meme, autre


# ==========================================================================
# 4. Export du dataset nettoyé
# ==========================================================================
def _export_one(args) -> str:
    src, dst, size = args
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    to_square(load_rgb(src), size, mode="pad").save(dst, quality=95)
    return dst


def export_clean(index: pd.DataFrame, out_dir: Path = IMAGES_CLEAN, size: int = IMG_SIZE,
                 n_jobs: int = 4) -> pd.DataFrame:
    """Écrit chaque image retenue en JPEG size x size (RGB, fond blanc, carré).

    Les images nettoyées sont ~10 fois plus rapides à charger pendant
    l'entraînement que les originaux (jusqu'à plusieurs milliers de pixels).
    """
    index = index.copy()
    index["clean_path"] = [str(Path(out_dir).relative_to(ROOT_DIR) / f"{n:03d}" / f"{i:05d}.jpg")
                           for i, n in zip(range(len(index)), index["pokedex_number"])]
    jobs = [(IMAGES_RAW.parent / p, ROOT_DIR / c, size) for p, c in zip(index["path"], index["clean_path"])]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        list(ex.map(_export_one, jobs, chunksize=32))
    return index


# ==========================================================================
# 5. Pipeline complet : téléchargement -> inventaire -> nettoyage -> export
# ==========================================================================
SEUIL_MEME_ESPECE = 8      # quasi-doublons d'une même espèce : distance de Hamming <= 8 bits sur 64
SEUIL_AUTRE_ESPECE = 4     # même image (ou presque) rangée dans 2 espèces : <= 4 bits
STD_MIN = 5.0              # image quasi uniforme (vide)


def download_dataset(dest: Path = IMAGES_RAW.parent) -> Path:
    """Télécharge et décompresse l'archive Kaggle (≈ 3 Go) si elle n'est pas déjà là.

    Le dataset est public : aucun compte n'est nécessaire pour l'URL de
    téléchargement direct. Sur une machine sans accès à kaggle.com, téléchargez
    l'archive à la main (bouton « Download » de la page Kaggle) et décompressez-la
    dans data/images/ (on doit obtenir data/images/pokemon/<NomDuPokémon>/…).
    """
    import urllib.request
    import zipfile
    if IMAGES_RAW.exists() and any(IMAGES_RAW.iterdir()):
        return IMAGES_RAW
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "archive.zip"
    print("Téléchargement de l'archive Kaggle (≈ 3 Go)…")
    urllib.request.urlretrieve(KAGGLE_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        suspects = [n for n in z.namelist() if n.startswith("/") or ".." in n.split("/")]
        if suspects:
            raise ValueError(f"Archive suspecte : {suspects[:3]}")
        z.extractall(dest)
    zpath.unlink()
    return IMAGES_RAW


def clean_index(scan: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    """Décide, pour chaque fichier, s'il est gardé ou retiré (colonne `motif`).

    Ordre des règles (la première qui s'applique donne le motif) :
      1. fichier illisible ou non image ;
      2. dossier non apparié au CSV ;
      3. image quasi vide ;
      4. image identique ou quasi identique présente dans PLUSIEURS espèces
         (étiquette ambiguë : on ne peut pas savoir laquelle est juste) ;
      5. doublon / quasi-doublon dans la même espèce : on garde la plus grande.
    """
    idx = scan.merge(assoc, on="folder", how="left")
    idx["motif"] = ""
    idx.loc[~idx["ok"], "motif"] = "illisible : " + idx["error"]
    idx.loc[(idx["motif"] == "") & idx["pokedex_number"].isna(), "motif"] = "dossier non apparié"
    idx.loc[(idx["motif"] == "") & (idx["std_pixels"] < STD_MIN), "motif"] = "image vide"

    cand = idx[idx["motif"] == ""].copy()
    h = np.array([int(x, 16) for x in cand["phash"]], dtype=np.uint64)
    esp = cand["pokedex_number"].to_numpy()
    md5 = cand["md5"].to_numpy()
    # 4. conflits entre espèces (même MD5 ou pHash très proche dans une autre espèce)
    conflit = np.zeros(len(cand), dtype=bool)
    md5_multi = cand.groupby("md5")["pokedex_number"].nunique()
    conflit |= np.isin(md5, md5_multi[md5_multi > 1].index)
    for i in range(len(h)):
        d = np.bitwise_count(h ^ h[i])
        if np.any((d <= SEUIL_AUTRE_ESPECE) & (esp != esp[i])):
            conflit[i] = True
    idx.loc[cand.index[conflit], "motif"] = "conflit entre espèces"

    # 5. groupes de quasi-doublons dans la même espèce (les doublons exacts ont d = 0)
    cand = idx[idx["motif"] == ""].copy()
    cand["groupe"] = hamming_matrix_groups(cand["phash"].tolist(), SEUIL_MEME_ESPECE,
                                           cand["pokedex_number"].to_numpy())
    cand["pixels"] = cand["width"] * cand["height"]
    garde = cand.sort_values(["groupe", "pixels"], ascending=[True, False]).drop_duplicates("groupe").index
    doublons = cand.index.difference(garde)
    idx.loc[doublons, "motif"] = np.where(cand.loc[doublons, "md5"].duplicated(keep=False) |
                                          cand.loc[doublons, "md5"].isin(cand.loc[garde, "md5"]),
                                          "doublon exact", "quasi-doublon")
    idx["groupe"] = cand["groupe"].reindex(idx.index)
    idx["garde"] = idx["motif"] == ""
    return idx


def prepare_dataset(n_jobs: int = 4) -> pd.DataFrame:
    """Pipeline complet, ré-exécutable : produit data/images_clean/ et data/images_index.csv."""
    from src.preprocessing import load_clean
    download_dataset()
    assoc, _, _ = match_folders(IMAGES_RAW, load_clean())
    idx = clean_index(scan_images(IMAGES_RAW, n_jobs), assoc)
    kept = export_clean(idx[idx["garde"]].sort_values(["pokedex_number", "path"]), n_jobs=n_jobs)
    idx = idx.merge(kept[["path", "clean_path"]], on="path", how="left")
    cols = ["path", "folder", "pokedex_number", "name", "type1", "type2", "garde", "motif", "groupe",
            "format", "mode", "width", "height", "n_frames", "transparency", "md5", "phash", "clean_path"]
    idx[cols].to_csv(IMAGES_INDEX, index=False)
    return idx[cols]


if __name__ == "__main__":
    out = prepare_dataset()
    print(out["motif"].replace("", "gardée").value_counts())


# ==========================================================================
# 6. Étiquettes et découpages (niveau 2)
# ==========================================================================
SPLITS_PATH = ROOT_DIR / "data" / "images_splits.csv"
# Types présents dans les images de 1re génération (Ténèbres n'existe pas en Gen 1)
TYPES_IMAGES = ["bug", "dragon", "electric", "fairy", "fighting", "fire", "flying", "ghost", "grass",
                "ground", "ice", "normal", "poison", "psychic", "rock", "steel", "water"]


def multilabel_matrix(df: pd.DataFrame, types: list[str] = TYPES_IMAGES) -> np.ndarray:
    """Matrice Y (n images x K types) : Y[i, k] = 1 si l'image i a le type k (type1 OU type2)."""
    Y = np.zeros((len(df), len(types)), dtype=np.float32)
    pos = {t: k for k, t in enumerate(types)}
    for i, (t1, t2) in enumerate(zip(df["type1"], df["type2"])):
        Y[i, pos[t1]] = 1
        if t2 in pos:
            Y[i, pos[t2]] = 1
    return Y


def make_splits(images: pd.DataFrame, seed: int = 42, n_folds: int = 5) -> pd.DataFrame:
    """Deux protocoles d'évaluation, figés une fois pour toutes :

    1. `split` : train / val / test = 70 / 15 / 15 %, STRATIFIÉ PAR POKÉMON
       (chaque espèce a ~70 % de ses images en train, ~15 % en val, ~15 % en test).
       Toutes les espèces sont vues à l'entraînement : protocole des approches A et B.
    2. `fold` : 5 plis StratifiedGroupKFold, GROUPÉS PAR ESPÈCE (toutes les images
       d'un Pokémon sont dans le même pli) et stratifiés sur type1. Évaluer sur
       le pli k un modèle entraîné sur les 4 autres = prédire les types de
       Pokémon JAMAIS VUS. Seule l'approche A (types) a un sens ici.
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    df = images[["path", "clean_path", "pokedex_number", "name", "type1", "type2"]].copy().reset_index(drop=True)
    y = df["pokedex_number"]
    i_train, i_tmp = train_test_split(df.index, test_size=0.30, stratify=y, random_state=seed)
    i_val, i_test = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=seed)
    df["split"] = ""
    df.loc[i_train, "split"], df.loc[i_val, "split"], df.loc[i_test, "split"] = "train", "val", "test"
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df["type1"], groups=df["pokedex_number"])):
        df.loc[te, "fold"] = k
    return df


def load_splits() -> pd.DataFrame:
    return pd.read_csv(SPLITS_PATH, keep_default_na=False)


# ==========================================================================
# 7. Caractéristiques « classiques » (référence sans réseau de neurones)
# ==========================================================================
CACHE_DIR = ROOT_DIR / "data" / "cache"
HIST_BINS = (12, 4, 4)       # teinte x saturation x valeur
HOG_SIZE = 64


def color_histogram(img: Image.Image, bins=HIST_BINS, white_thresh: int = 235) -> np.ndarray:
    """Histogramme de couleurs HSV normalisé, calculé SUR LE POKÉMON seulement.

    Les pixels presque blancs (fond ajouté au nettoyage ou fond d'origine)
    sont exclus : sinon le fond dominerait l'histogramme. HSV sépare la teinte
    (H, « quelle couleur »), la saturation (S, « couleur vive ou terne ») et la
    valeur (V, « clair ou sombre ») : c'est plus proche de la perception que RGB.
    """
    rgb = np.asarray(img.convert("RGB").resize((96, 96)))
    mask = ~(rgb.min(axis=2) >= white_thresh)
    hsv = np.asarray(img.convert("RGB").resize((96, 96)).convert("HSV")).reshape(-1, 3)[mask.ravel()]
    if len(hsv) < 20:                                   # image presque entièrement blanche
        hsv = np.asarray(img.convert("RGB").resize((96, 96)).convert("HSV")).reshape(-1, 3)
    h, _ = np.histogramdd(hsv, bins=bins, range=((0, 256), (0, 256), (0, 256)))
    h = h.ravel()
    return (h / h.sum()).astype(np.float32)


def hog_features(img: Image.Image, size: int = HOG_SIZE) -> np.ndarray:
    """Histogramme de gradients orientés (HOG) : décrit la FORME (contours).

    L'image en niveaux de gris est découpée en cellules de 8 x 8 pixels ;
    dans chaque cellule on calcule l'histogramme des directions des gradients
    (9 orientations), puis on normalise par blocs de 2 x 2 cellules
    (robustesse aux variations d'éclairage).
    """
    from skimage.feature import hog
    g = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32) / 255.0
    return hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def _classic_one(path: str) -> np.ndarray:
    img = Image.open(ROOT_DIR / path).convert("RGB")
    return np.concatenate([color_histogram(img), hog_features(img)])


def classic_features(paths, n_jobs: int = 4, cache: str | None = "classic") -> np.ndarray:
    """Matrice (n images x 1 956) : 192 bins de couleur + 1 764 valeurs HOG, mise en cache."""
    cache_file = CACHE_DIR / f"{cache}.npz" if cache else None
    paths = list(paths)
    if cache_file and cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        if list(z["paths"]) == paths:
            return z["X"]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        X = np.stack(list(ex.map(_classic_one, paths, chunksize=64)))
    if cache_file:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=X, paths=np.array(paths, dtype=object))
    return X


# ==========================================================================
# 8. Métriques (communes à tous les modèles d'images)
# ==========================================================================
def decide_types(P: np.ndarray, thresholds=0.5, max_types: int = 2) -> np.ndarray:
    """Probabilités (n x K) -> décision 0/1 : p >= seuil, au moins 1 type, au plus 2."""
    P = np.asarray(P)
    D = (P >= np.broadcast_to(thresholds, P.shape[1:])).astype(int)
    top = np.argsort(-P, axis=1)
    D[np.arange(len(P)), top[:, 0]] = 1                         # au moins le plus probable
    for i in np.nonzero(D.sum(1) > max_types)[0]:                # au plus max_types
        D[i] = 0
        D[i, top[i, :max_types]] = 1
    return D


def multilabel_scores(Y: np.ndarray, D: np.ndarray) -> dict:
    """Y, D : matrices 0/1 (n x K), vérité et décision."""
    from sklearn.metrics import f1_score, hamming_loss
    exact = (Y == D).all(axis=1)
    return {"F1 micro": f1_score(Y, D, average="micro", zero_division=0),
            "F1 macro": f1_score(Y, D, average="macro", zero_division=0),
            "Hamming loss": hamming_loss(Y, D),
            "exact match": exact.mean(),
            "≥ 1 type correct": ((Y * D).sum(1) > 0).mean()}


def species_to_types(P_species: np.ndarray, species: np.ndarray, type_table: pd.DataFrame,
                     types: list[str] = TYPES_IMAGES) -> np.ndarray:
    """Approche B -> probabilités de types : P(type k) = somme des P(espèce s) pour s ayant k."""
    M = multilabel_matrix(type_table.set_index("pokedex_number").loc[species].reset_index(), types)
    return P_species @ M          # (n x S) @ (S x K)


def species_scores(y: np.ndarray, P: np.ndarray, classes: np.ndarray) -> dict:
    from sklearn.metrics import f1_score
    order = np.argsort(-P, axis=1)
    pred = classes[order[:, 0]]
    top5 = (classes[order[:, :5]] == np.asarray(y)[:, None]).any(axis=1)
    return {"top-1": (pred == y).mean(), "top-5": top5.mean(),
            "F1 macro (espèces)": f1_score(y, pred, average="macro", zero_division=0)}


def coverage(correct: np.ndarray, species: np.ndarray) -> tuple[pd.Series, dict]:
    """Couverture : réussite calculée POUR CHAQUE POKÉMON, puis part des Pokémon
    dont la réussite dépasse 50 / 70 / 90 %."""
    acc = pd.Series(np.asarray(correct, dtype=float)).groupby(np.asarray(species)).mean()
    return acc, {f"% Pokémon > {s} %": (acc > s / 100).mean() for s in (50, 70, 90)}
