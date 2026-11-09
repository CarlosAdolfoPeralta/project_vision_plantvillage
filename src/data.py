"""Carga y preparación de datos para el proyecto PlantVillage (Keras)."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import GroupShuffleSplit, train_test_split

_ROOT = Path(__file__).resolve().parent.parent

# Si existe la carpeta local data/plant_village se lee desde ahí;
# si no, el dataset se descarga desde Hugging Face.
_LOCAL = _ROOT / "data" / "plant_village"
REPO = str(_LOCAL) if _LOCAL.exists() else "DScomp380/plant_village"

CACHE_DIR = _ROOT / "data" / "cache"
IMG_SIZE = (224, 224)  # (ancho, alto)
SEED = 42


def load_splits(repo=REPO, val_size=0.15, test_size=0.15):
    """Devuelve (train, val, test) como datasets de Hugging Face.

    - Si el dataset trae `leaf_id` y un split `test`, usa el split oficial y
      arma la validación agrupando por hoja (sin fuga).
    - Si no (caso de DScomp380/plant_village), hace un split estratificado
      por clase. En ese caso puede haber casi-duplicados entre splits (fotos de
      una misma hoja), lo que hace que las métricas sean algo optimistas.
    """
    ds = load_dataset(repo)
    first = ds[list(ds.keys())[0]]

    if "leaf_id" in first.column_names and "test" in ds:
        groups = np.array(ds["train"]["leaf_id"])
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
        tr_idx, va_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
        return ds["train"].select(tr_idx), ds["train"].select(va_idx), ds["test"]

    print(
        "WARNING: the dataset does not include leaf_id; a class-stratified split is used. "
        "There may be leakage between images of the same leaf, so the metrics "
        "may be somewhat optimistic."
    )
    full = concatenate_datasets([ds[s] for s in ds.keys()])
    y = np.array(full["label"])
    idx = np.arange(len(full))
    rest = val_size + test_size

    tr_idx, tmp_idx = train_test_split(
        idx, test_size=rest, stratify=y, random_state=SEED
    )
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=test_size / rest, stratify=y[tmp_idx], random_state=SEED
    )
    return full.select(tr_idx), full.select(va_idx), full.select(te_idx)


def to_arrays(hf_ds, img_size=IMG_SIZE, name=None, verbose=True):
    """Convierte un dataset de Hugging Face en arrays (X uint8, y int64).

    Decodificar y redimensionar JPGs en cada época es lo más lento en CPU, así
    que se hace una sola vez. Si se pasa `name`, el resultado se guarda en
    data/cache/ y las próximas llamadas lo leen directo del disco.
    """
    w, h = img_size
    n = len(hf_ds)

    cache_file = None
    if name is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"{name}_{n}_{w}x{h}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            if verbose:
                print(f"[cache] {cache_file.name}")
            return data["X"], data["y"]

    X = np.empty((n, h, w, 3), dtype=np.uint8)
    y = np.empty(n, dtype=np.int64)
    for i, ex in enumerate(hf_ds):
        X[i] = np.asarray(ex["image"].convert("RGB").resize((w, h)), dtype=np.uint8)
        y[i] = ex["label"]
        if verbose and (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n} processed images")

    if cache_file is not None:
        np.savez(cache_file, X=X, y=y)
    return X, y


def to_tf(hf_ds, batch_size=32, shuffle=False, img_size=IMG_SIZE):
    """Convierte un dataset de Hugging Face en un tf.data.Dataset (lazy, más lento en CPU)."""
    w, h = img_size

    def gen():
        for ex in hf_ds:
            img = ex["image"].convert("RGB").resize((w, h))
            yield np.asarray(img, dtype=np.uint8), ex["label"]

    tf_ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(h, w, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )
    if shuffle:
        tf_ds = tf_ds.shuffle(1000, seed=SEED)
    return tf_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
