"""Extracción de features con redes preentrenadas (transfer learning en CPU)."""
import time

import numpy as np
from tensorflow import keras

from .data import CACHE_DIR


def build_backbone(name="mobilenetv3small", img_size=(224, 224)):
    """Red preentrenada en ImageNet, sin la capa final, con pooling global.

    Las dos opciones incluyen su propio preprocesado: reciben píxeles 0-255.
    """
    w, h = img_size
    kwargs = dict(
        input_shape=(h, w, 3), include_top=False, weights="imagenet", pooling="avg"
    )
    if name == "mobilenetv3small":
        return keras.applications.MobileNetV3Small(**kwargs)
    if name == "efficientnetb0":
        return keras.applications.EfficientNetB0(**kwargs)
    raise ValueError(f"Backbone no soportado: {name}")


def extract_features(hf_ds, backbone, img_size=(224, 224), name=None,
                     batch_size=64, verbose=True):
    """Pasa cada imagen UNA vez por el backbone y devuelve (features, labels).

    Procesa por lotes (no guarda las imágenes en memoria, solo los vectores) y,
    si se pasa `name`, cachea el resultado en data/cache/.
    """
    w, h = img_size
    n = len(hf_ds)

    cache_file = None
    if name is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = CACHE_DIR / f"feat_{name}_{backbone.name}_{n}_{w}x{h}.npz"
        if cache_file.exists():
            if verbose:
                print(f"[cache] {cache_file.name}")
            data = np.load(cache_file)
            return data["X"], data["y"]

    feats, labels = [], []
    t0 = time.time()
    done = 0
    for i, batch in enumerate(hf_ds.iter(batch_size=batch_size)):
        imgs = np.stack(
            [np.asarray(im.convert("RGB").resize((w, h)), dtype=np.uint8)
             for im in batch["image"]]
        )
        feats.append(backbone.predict(imgs, verbose=0))
        labels.extend(batch["label"])
        done += len(imgs)
        if verbose and (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {done}/{n} imágenes | {el / 60:.1f} min | "
                  f"{done / el:.0f} img/s | faltan ~{(n - done) / (done / el) / 60:.1f} min")

    X = np.concatenate(feats).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if cache_file is not None:
        np.savez(cache_file, X=X, y=y)
    return X, y
