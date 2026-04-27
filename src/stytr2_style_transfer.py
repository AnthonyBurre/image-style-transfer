"""StyTr² style transfer (Deng et al., CVPR 2022).

Transformer-based arbitrary style transfer. Architecturally distinct from
Magenta (no instance-norm modulation by a global style code) and from Gatys
(no per-image optimisation loop): a transformer cross-attends content patches
to style patches, then a CNN decoder maps patch embeddings back to pixels.

Pretrained weights are mirrored on Hugging Face under
``datnguyentien204/Sty_TR2_38`` (the original repo only ships them on Google
Drive). Total ~141 MB across three files, downloaded into ``./model-stytr2/``
on first use and reused thereafter. The fourth upstream file
(``vgg_normalised.pth``, 80 MB) is only used by the training-time loss and is
deliberately skipped here.
"""
import os
from collections import OrderedDict

import numpy as np
import requests
from PIL import Image, ImageOps

# Torch import is local to keep the Magenta-only path free of the ~800 MB
# PyTorch dependency cost on startup.
_inference_fn = None

# StyTr-2 was trained on 256x256 patches at patch_size=8 (32x32 tokens).
# At inference the model handles arbitrary square sizes since PatchEmbed is a
# stride-8 conv, but attention is O(N^2) in the patch count, so 512x512
# (64x64 = 4096 tokens) is roughly the ceiling on a 4 GB CPU container.
INPUT_SIZE = 512

WEIGHT_DIR = "model-stytr2"
WEIGHT_REPO = "datnguyentien204/Sty_TR2_38"
WEIGHT_FILES = (
    "decoder_iter_160000.pth",
    "transformer_iter_160000.pth",
    "embedding_iter_160000.pth",
)


def _hf_url(filename):
    return f"https://huggingface.co/{WEIGHT_REPO}/resolve/main/experiments/{filename}"


def _download_weights(weight_dir=WEIGHT_DIR):
    os.makedirs(weight_dir, exist_ok=True)
    for filename in WEIGHT_FILES:
        dest = os.path.join(weight_dir, filename)
        if os.path.exists(dest):
            continue
        url = _hf_url(filename)
        print(f"Downloading {filename} from {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    return weight_dir


def _load_state_dict(path):
    # The released checkpoints were saved as plain state dicts with no
    # ``module.`` prefix, so we can pass them through unchanged. Older copies
    # sometimes had the prefix; strip it defensively.
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=True)
    cleaned = OrderedDict()
    for k, v in raw.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def _build_inference():
    import torch

    from .stytr2.model import PatchEmbed, StyTransInference, decoder
    from .stytr2.transformer import Transformer

    weight_dir = _download_weights()

    embedding = PatchEmbed()
    transformer = Transformer()

    decoder.load_state_dict(_load_state_dict(os.path.join(weight_dir, "decoder_iter_160000.pth")))
    transformer.load_state_dict(_load_state_dict(os.path.join(weight_dir, "transformer_iter_160000.pth")))
    embedding.load_state_dict(_load_state_dict(os.path.join(weight_dir, "embedding_iter_160000.pth")))

    network = StyTransInference(embedding, transformer, decoder)
    network.eval()

    @torch.no_grad()
    def run(content_tensor, style_tensor):
        return network(content_tensor, style_tensor)

    return run


def _get_inference():
    global _inference_fn
    if _inference_fn is None:
        print("Loading StyTr² model (first run downloads ~221 MB of weights)...")
        _inference_fn = _build_inference()
        print("StyTr² loaded.")
    return _inference_fn


def _pil_to_tensor(image, size):
    """PIL -> 1xCxHxW float32 [0,1] tensor, square-cropped to ``size``.

    StyTr-2's transformer collapses the patch grid via ``H = int(sqrt(N))``,
    which only round-trips correctly when the feature map is square. So we
    follow the upstream test script: resize the shortest side to ``size``,
    then centre-crop to ``size``x``size``.
    """
    import torch

    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    short = min(w, h)
    scale = size / short
    image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)

    new_w, new_h = image.size
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _tensor_to_pil(tensor):
    arr = tensor.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def perform_stytr2_style_transfer(content_image, style_image):
    if content_image is None or style_image is None:
        return None

    run = _get_inference()

    print("Processing images...")
    content_tensor = _pil_to_tensor(content_image, INPUT_SIZE)
    style_tensor = _pil_to_tensor(style_image, INPUT_SIZE)

    print("Applying StyTr² style transfer...")
    output = run(content_tensor, style_tensor)

    print("Conversion complete. Returning final image.")
    return _tensor_to_pil(output)
