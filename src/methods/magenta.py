"""Magenta arbitrary-image-stylization-v1-256 (TF Hub).

Feed-forward inference: the network takes a content tensor and a style tensor
and returns the stylised image in one shot. Weights are downloaded as a
tarball into ``./model/`` on first use and reused thereafter.
"""
import os
import tarfile

import numpy as np
import requests
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from ..image import prepare

LABEL = "Magenta (fast — ~seconds)"
BLURB = "**Magenta** is a feed-forward model that returns in seconds"
REQUIRES_SQUARE = False

MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed"
MODEL_DIR = "model"

CONTENT_MAX_DIM = 2048
STYLE_MAX_DIM = 256

_model = None


def _download_model(model_dir=MODEL_DIR):
    # Existence check is purely directory-presence; a partial extraction must
    # be cleaned up by deleting ``model_dir`` to force a re-download.
    if os.path.exists(model_dir):
        return model_dir

    print(f"Downloading Magenta model from {MODEL_URL}...")
    os.makedirs(model_dir, exist_ok=True)

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    archive = os.path.join(model_dir, "model.tar.gz")
    with open(archive, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with tarfile.open(archive) as tar:
        # filter='data' (PEP 706) blocks path traversal and unsafe members.
        tar.extractall(path=model_dir, filter="data")

    os.remove(archive)
    return model_dir


def _get_model():
    global _model
    if _model is None:
        print("Loading Magenta model...")
        _model = hub.load(_download_model())
        print("Magenta loaded.")
    return _model


def _to_tensor(image, max_dim):
    arr = np.asarray(prepare(image, max_dim), dtype=np.float32) / 255.0
    return tf.convert_to_tensor(arr)[tf.newaxis, :]


def _to_pil(tensor):
    arr = (np.array(tensor) * 255.0).astype(np.uint8)
    if arr.ndim > 3:
        arr = arr[0]
    return Image.fromarray(arr)


def stylize(content_image, style_image, *, progress=None):
    if content_image is None or style_image is None:
        return None
    model = _get_model()
    content_tensor = _to_tensor(content_image, CONTENT_MAX_DIM)
    style_tensor = _to_tensor(style_image, STYLE_MAX_DIM)
    output = model(content_tensor, style_tensor)[0]
    return _to_pil(output)
