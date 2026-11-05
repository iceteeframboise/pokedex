"""Entraînement des réseaux de neurones sur les images (niveau 2).

Usage (depuis la racine du projet) :
    python -m src.train_images --model cnn_scratch --approach B
    python -m src.train_images --model cnn_scratch --approach A

Chaque entraînement écrit :
    models/candidates/<modèle>_<approche>.keras         le réseau (meilleure époque)
    reports/tables/histories/<modèle>_<approche>.json   courbes d'apprentissage + durées
    data/cache/preds/<modèle>_<approche>_val.npy        probabilités sur la validation

Le choix du modèle final se fait dans le notebook 2 ; seul le test final
(partie 5) touche au jeu de test.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import RANDOM_STATE, ROOT_DIR
from src.image_utils import CACHE_DIR, TYPES_IMAGES, load_splits, multilabel_matrix

CANDIDATES = ROOT_DIR / "models" / "candidates"
HISTORIES = ROOT_DIR / "reports" / "tables" / "histories"
PREDS = CACHE_DIR / "preds"
WHITE = 255.0


# ==========================================================================
# 1. Données en mémoire
# ==========================================================================
def load_arrays(size: int, paths=None) -> np.ndarray:
    """Toutes les images nettoyées en uint8 (n, size, size, 3), mises en cache.

    14 715 images en 96 x 96 = 400 Mo : tout tient en mémoire, ce qui évite de
    décoder les JPEG à chaque époque.
    """
    splits = load_splits()
    paths = list(splits["clean_path"] if paths is None else paths)
    f = CACHE_DIR / f"images_{size}.npy"
    if f.exists() and paths == list(splits["clean_path"]):
        return np.load(f)
    X = np.stack([np.asarray(Image.open(ROOT_DIR / p).convert("RGB").resize((size, size), Image.LANCZOS))
                  for p in paths]).astype(np.uint8)
    if paths == list(splits["clean_path"]):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(f, X)
    return X


def targets(splits, approach: str):
    """A : matrice multi-label (n x 17). B : indice d'espèce (0..141) + table des classes."""
    if approach == "A":
        return multilabel_matrix(splits), np.array(TYPES_IMAGES)
    classes = np.sort(splits["pokedex_number"].unique())
    return np.searchsorted(classes, splits["pokedex_number"].to_numpy()), classes


# ==========================================================================
# 2. Augmentation de données
# ==========================================================================
def augmentation(seed: int = RANDOM_STATE) -> keras.Sequential:
    """Transformations aléatoires appliquées au TRAIN uniquement, à chaque époque.

    Le modèle voit ainsi des variantes plausibles de chaque image (cadrage,
    orientation, éclairage, netteté) et apprend à ne pas en dépendre : c'est
    une régularisation, qui réduit le surapprentissage. On complète les zones
    vides par du BLANC (même fond que le nettoyage).

    Pas de changement de teinte : la couleur est un indice essentiel du type
    (un Carapuce rose n'existe pas) ; on ne modifie que légèrement la
    saturation et la luminosité.
    """
    L = keras.layers
    return keras.Sequential([
        L.RandomFlip("horizontal", seed=seed),
        L.RandomRotation(0.06, fill_mode="constant", fill_value=WHITE, seed=seed),     # ± 22°
        L.RandomZoom((-0.15, 0.15), fill_mode="constant", fill_value=WHITE, seed=seed),
        L.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=WHITE, seed=seed),
        L.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
        L.RandomContrast(0.15, seed=seed),
        L.RandomSaturation((0.4, 0.6), value_range=(0, 255), seed=seed),               # 0,5 = inchangé
        L.RandomGaussianBlur(factor=0.3, kernel_size=3, sigma=1.0, value_range=(0, 255), seed=seed),
    ], name="augmentation")


def make_dataset(X, y, batch: int, training: bool, seed: int = RANDOM_STATE):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    if training:
        aug = augmentation(seed)
        ds = ds.map(lambda a, b: (tf.clip_by_value(aug(tf.cast(a, tf.float32), training=True), 0, 255), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda a, b: (tf.cast(a, tf.float32), b))
    return ds.prefetch(tf.data.AUTOTUNE)


# ==========================================================================
# 3. Fonctions de coût pondérées
# ==========================================================================
@keras.saving.register_keras_serializable(package="pokedex")
class WeightedBCE(keras.losses.Loss):
    """Entropie croisée binaire avec un poids w_k sur les EXEMPLES POSITIFS du type k.

        L = -1/K Σ_k [ w_k · y_k · ln p_k + (1 - y_k) · ln(1 - p_k) ]

    w_k = sqrt(n_négatifs_k / n_positifs_k) : un type rare (Acier, 1 % des
    images) compte davantage, sans que le poids explose (racine carrée : un
    poids égal au rapport complet, ~80, rendrait l'entraînement instable et
    gonflerait les probabilités, cf. niveau 1).
    """

    def __init__(self, pos_weight=None, name="weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weight = list(map(float, pos_weight)) if pos_weight is not None else None

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        w = tf.constant(self.pos_weight, dtype=p.dtype) if self.pos_weight else 1.0
        return -tf.reduce_mean(w * y_true * tf.math.log(p) + (1 - y_true) * tf.math.log(1 - p), axis=-1)

    def get_config(self):
        return {**super().get_config(), "pos_weight": self.pos_weight}


def class_weights_species(y_idx: np.ndarray) -> dict:
    """Poids n / (S · n_s) par espèce : chaque espèce pèse autant au total."""
    counts = np.bincount(y_idx)
    return {i: len(y_idx) / (len(counts) * c) for i, c in enumerate(counts)}


# ==========================================================================
# 4. Architectures
# ==========================================================================
def build_scratch_cnn(size: int, n_out: int, approach: str, filters=(24, 48, 96, 192),
                      dropout: float = 0.4) -> keras.Model:
    """CNN « from scratch » : 4 blocs [Conv 3x3 - BN - ReLU] x 2 + MaxPool.

    - convolution 3 x 3 : chaque filtre détecte un motif local (bord, coin,
      texture) ; en empilant les blocs, le champ récepteur grandit et les
      motifs deviennent des parties (œil, aile, flamme) ;
    - BatchNormalization : normalise les activations de chaque couche
      (moyenne 0, variance 1 sur le batch), ce qui stabilise et accélère
      l'apprentissage ;
    - MaxPooling 2 x 2 : divise la résolution par 2 (invariance aux petites
      translations, moins de calculs) ;
    - GlobalAveragePooling : moyenne de chaque carte de caractéristiques ->
      un vecteur de taille fixe, beaucoup moins de paramètres qu'un Flatten ;
    - Dropout : éteint aléatoirement 40 % des neurones à l'entraînement
      (régularisation).
    Sortie : 17 sigmoïdes (A) ou softmax sur 142 espèces (B).
    """
    L = keras.layers
    inp = keras.Input((size, size, 3))
    x = L.Rescaling(1 / 255.0)(inp)
    for f in filters:
        for _ in range(2):
            x = L.Conv2D(f, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(n_out, activation="sigmoid" if approach == "A" else "softmax", name="sortie")(x)
    return keras.Model(inp, out, name=f"cnn_scratch_{approach}")


MODELS = {
    "cnn_scratch": dict(builder=build_scratch_cnn, size=96, batch=64, lr=2e-3, epochs=40, patience=6),
}


# ==========================================================================
# 5. Entraînement
# ==========================================================================
def train(model_key: str, approach: str, verbose: int = 2, **overrides) -> dict:
    cfg = {**MODELS[model_key], **overrides}
    keras.utils.set_random_seed(RANDOM_STATE)
    splits = load_splits()
    X = load_arrays(cfg["size"])
    y, classes = targets(splits, approach)
    tr = (splits["split"] == "train").to_numpy()
    va = (splits["split"] == "val").to_numpy()

    model = cfg["builder"](cfg["size"], len(classes), approach)
    if approach == "A":
        pos = y[tr].mean(0)
        loss = WeightedBCE(np.sqrt((1 - pos) / pos))
        metrics = [keras.metrics.BinaryAccuracy(name="acc", threshold=0.5)]
        class_weight = None
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc"),
                   keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")]
        class_weight = class_weights_species(y[tr])
    model.compile(optimizer=keras.optimizers.AdamW(cfg["lr"], weight_decay=1e-4), loss=loss, metrics=metrics)

    callbacks = [
        # arrêt anticipé : on s'arrête quand la loss de VALIDATION ne s'améliore plus,
        # et on revient aux poids de la meilleure époque
        keras.callbacks.EarlyStopping("val_loss", patience=cfg["patience"], restore_best_weights=True),
        # pas d'apprentissage divisé par 2 quand la validation stagne
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-5),
        # sauvegarde à chaque époque : si le processus est interrompu, relancer la même
        # commande reprend l'entraînement là où il s'était arrêté
        keras.callbacks.BackupAndRestore(str(CACHE_DIR / "backup" / f"{model_key}_{approach}"),
                                         delete_checkpoint=True),
    ]
    t0 = time.time()
    hist = model.fit(make_dataset(X[tr], y[tr], cfg["batch"], True),
                     validation_data=make_dataset(X[va], y[va], cfg["batch"], False),
                     epochs=cfg["epochs"], class_weight=class_weight, callbacks=callbacks, verbose=verbose)
    train_time = time.time() - t0

    t0 = time.time()
    P_val = model.predict(make_dataset(X[va], y[va], 128, False), verbose=0)
    infer_ms = (time.time() - t0) / va.sum() * 1000

    name = f"{model_key}_{approach}"
    for d in (CANDIDATES, HISTORIES, PREDS):
        d.mkdir(parents=True, exist_ok=True)
    model.save(CANDIDATES / f"{name}.keras")
    np.save(PREDS / f"{name}_val.npy", P_val)
    info = {"model": model_key, "approach": approach, "config": {k: v for k, v in cfg.items() if k != "builder"},
            "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
            "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1, "epochs_run": len(hist.history["loss"]),
            "train_time_s": train_time, "infer_ms_per_image": infer_ms, "params": int(model.count_params()),
            "size_mb": (CANDIDATES / f"{name}.keras").stat().st_size / 1e6, "classes": classes.tolist()}
    (HISTORIES / f"{name}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


# ==========================================================================
# 6. Transfer learning (réseaux pré-entraînés sur ImageNet)
# ==========================================================================
def _resnet_preprocess(x):
    return keras.applications.resnet50.preprocess_input(x)


# Chaque réseau attend des pixels 0-255 : MobileNetV3 et EfficientNet intègrent
# leur propre normalisation ; ResNet50 a besoin de preprocess_input (BGR, moyenne ImageNet).
BACKBONES = {
    "mobilenetv3_small": (lambda: keras.applications.MobileNetV3Small(include_top=False, weights="imagenet",
                                                                     input_shape=(224, 224, 3), pooling="avg"), None),
    "mobilenetv3_large": (lambda: keras.applications.MobileNetV3Large(include_top=False, weights="imagenet",
                                                                     input_shape=(224, 224, 3), pooling="avg"), None),
    "efficientnet_b0": (lambda: keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                                 input_shape=(224, 224, 3), pooling="avg"), None),
    "resnet50": (lambda: keras.applications.ResNet50(include_top=False, weights="imagenet",
                                                     input_shape=(224, 224, 3), pooling="avg"), _resnet_preprocess),
}


def build_backbone(name: str) -> keras.Model:
    """Réseau pré-entraîné SANS sa couche de classification ImageNet, suivi
    d'un GlobalAveragePooling : image 224 x 224 -> vecteur de caractéristiques."""
    make, prep = BACKBONES[name]
    base = make()
    inp = keras.Input((224, 224, 3))
    x = keras.layers.Lambda(prep, name="preprocess")(inp) if prep else inp
    return keras.Model(inp, base(x), name=f"{name}_features")


def extract_embeddings(name: str, batch: int = 64) -> np.ndarray:
    """Caractéristiques du réseau GELÉ pour les 14 715 images (calculées une fois, mises en cache)."""
    f = CACHE_DIR / f"emb_{name}.npy"
    if f.exists():
        return np.load(f)
    X = load_arrays(224)
    model = build_backbone(name)
    t0 = time.time()
    E = model.predict(tf.data.Dataset.from_tensor_slices(X).batch(batch).map(lambda a: tf.cast(a, tf.float32)),
                      verbose=0)
    (CACHE_DIR / f"emb_{name}.time").write_text(f"{(time.time() - t0) / len(X) * 1000:.2f}")
    np.save(f, E.astype(np.float32))
    return E


def build_head(dim: int, n_out: int, approach: str, dropout: float = 0.3) -> keras.Model:
    """Tête de classification posée sur les caractéristiques : Dropout + Dense."""
    inp = keras.Input((dim,))
    x = keras.layers.Dropout(dropout)(inp)
    out = keras.layers.Dense(n_out, activation="sigmoid" if approach == "A" else "softmax", name="sortie")(x)
    return keras.Model(inp, out, name=f"tete_{approach}")


def compile_for(model, approach, y_train, lr):
    if approach == "A":
        pos = y_train.mean(0)
        loss = WeightedBCE(np.sqrt((1 - pos) / np.clip(pos, 1e-6, None)))
        metrics = [keras.metrics.BinaryAccuracy(name="acc")]
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc"),
                   keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")]
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=1e-4), loss=loss, metrics=metrics)


def train_head(E: np.ndarray, y: np.ndarray, approach: str, tr: np.ndarray, va: np.ndarray,
               epochs: int = 200, lr: float = 1e-3, verbose: int = 0):
    """Entraîne la tête sur des caractéristiques gelées (quelques secondes sur CPU)."""
    keras.utils.set_random_seed(RANDOM_STATE)
    head = build_head(E.shape[1], y.shape[1] if approach == "A" else int(y.max()) + 1, approach)
    compile_for(head, approach, y[tr], lr)
    cw = class_weights_species(y[tr]) if approach == "B" else None
    hist = head.fit(E[tr], y[tr], validation_data=(E[va], y[va]), epochs=epochs, batch_size=128,
                    class_weight=cw, verbose=verbose,
                    callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)])
    return head, hist


def finetune(backbone: str, approach: str, n_trainable: int = 40, epochs: int = 12, lr: float = 1e-4,
             batch: int = 32, verbose: int = 2) -> dict:
    """Fine-tuning : on part d'une tête déjà entraînée sur les caractéristiques
    gelées, puis on DÉGÈLE les `n_trainable` dernières couches du réseau et on
    réentraîne l'ensemble avec un petit pas d'apprentissage (1e-4) et la data
    augmentation. Les BatchNorm restent gelées (statistiques d'ImageNet), sinon
    de petits batchs les dérègleraient.
    """
    keras.utils.set_random_seed(RANDOM_STATE)
    splits = load_splits()
    y, classes = targets(splits, approach)
    tr = (splits["split"] == "train").to_numpy()
    va = (splits["split"] == "val").to_numpy()
    E = extract_embeddings(backbone)
    head, _ = train_head(E, y, approach, tr, va)                         # 1) tête sur réseau gelé

    feat = build_backbone(backbone)
    base = [l for l in feat.layers if isinstance(l, keras.Model)][0]
    base.trainable = True
    for layer in base.layers[:-n_trainable]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    inp = keras.Input((224, 224, 3))
    model = keras.Model(inp, head(feat(inp, training=False)), name=f"{backbone}_{approach}")
    compile_for(model, approach, y[tr], lr)

    X = load_arrays(224)
    name = f"{backbone}_ft_{approach}"
    callbacks = [keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-6),
                 keras.callbacks.BackupAndRestore(str(CACHE_DIR / "backup" / name), delete_checkpoint=True)]
    t0 = time.time()
    hist = model.fit(make_dataset(X[tr], y[tr], batch, True), validation_data=make_dataset(X[va], y[va], 64, False),
                     epochs=epochs, class_weight=class_weights_species(y[tr]) if approach == "B" else None,
                     callbacks=callbacks, verbose=verbose)
    train_time = time.time() - t0
    t0 = time.time()
    P_val = model.predict(make_dataset(X[va], y[va], 64, False), verbose=0)
    infer_ms = (time.time() - t0) / va.sum() * 1000
    for d in (CANDIDATES, HISTORIES, PREDS):
        d.mkdir(parents=True, exist_ok=True)
    model.save(CANDIDATES / f"{name}.keras")
    np.save(PREDS / f"{name}_val.npy", P_val)
    info = {"model": backbone, "approach": approach, "n_trainable": n_trainable, "lr": lr,
            "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
            "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1, "epochs_run": len(hist.history["loss"]),
            "train_time_s": train_time, "infer_ms_per_image": infer_ms, "params": int(model.count_params()),
            "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
            "size_mb": (CANDIDATES / f"{name}.keras").stat().st_size / 1e6, "classes": classes.tolist()}
    (HISTORIES / f"{name}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


# ==========================================================================
# 7. Grad-CAM : où le réseau regarde-t-il ?
# ==========================================================================
def grad_cam(model: keras.Model, images: np.ndarray, class_idx=None, layer: str = "top_activation") -> np.ndarray:
    """Cartes Grad-CAM (Selvaraju et al., 2017) pour un modèle [réseau pré-entraîné -> tête].

    1. On récupère les cartes d'activation A^k de la dernière couche convolutive
       (7 x 7 x 1 280 pour EfficientNetB0) et le score y^c de la classe c.
    2. Poids de chaque carte : alpha_k = moyenne spatiale de dy^c / dA^k
       (« à quel point la carte k fait monter le score de c »).
    3. Carte de chaleur : ReLU(sum_k alpha_k A^k), normalisée dans [0, 1] puis
       agrandie à la taille de l'image. Les zones chaudes sont celles qui ont
       le plus contribué à la décision.
    """
    feat, head = model.layers[1], model.layers[2]
    base = [l for l in feat.layers if isinstance(l, keras.Model)][0]
    conv_model = keras.Model(base.inputs, [base.get_layer(layer).output, base.output])
    x = tf.cast(images, tf.float32)
    with tf.GradientTape() as tape:
        conv, f = conv_model(x, training=False)
        tape.watch(conv)
        preds = head(f, training=False)
        idx = tf.argmax(preds, axis=1) if class_idx is None else tf.constant(np.asarray(class_idx), tf.int64)
        score = tf.gather(preds, idx, axis=1, batch_dims=1)
    grads = tape.gradient(score, conv)
    alpha = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(alpha * conv, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    cam = tf.image.resize(cam[..., None], images.shape[1:3], method="bilinear")[..., 0]
    return cam.numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_scratch", choices=list(MODELS))
    ap.add_argument("--approach", default="B", choices=["A", "B"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--finetune", help="nom du réseau pré-entraîné à affiner (ex. mobilenetv3_large)")
    ap.add_argument("--n-trainable", type=int, default=40)
    a = ap.parse_args()
    if a.finetune:
        info = finetune(a.finetune, a.approach, n_trainable=a.n_trainable, **({"epochs": a.epochs} if a.epochs else {}))
        print(f"{a.finetune} fine-tuning {a.approach} : meilleure époque {info['best_epoch']}/{info['epochs_run']}, "
              f"{info['train_time_s'] / 60:.1f} min")
        raise SystemExit
    info = train(a.model, a.approach, **({"epochs": a.epochs} if a.epochs else {}))
    print(f"{a.model} {a.approach} : meilleure époque {info['best_epoch']}/{info['epochs_run']}, "
          f"{info['train_time_s'] / 60:.1f} min, {info['params']} paramètres")
