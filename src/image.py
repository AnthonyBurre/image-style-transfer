"""PIL helpers shared by method modules.

Each method then wraps the returned PIL image into its own framework tensor
(TF for Magenta/Gatys, torch for StyTr²).
"""
from PIL import Image, ImageOps


def prepare(image, max_dim):
    """EXIF-corrected RGB PIL image, longest side ≤ ``max_dim``.

    LANCZOS keeps edges sharper than bilinear on large downsamples. Smaller
    images are not upscaled.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image).__name__}")

    image = ImageOps.exif_transpose(image).convert("RGB")
    long_dim = max(image.size)
    if long_dim > max_dim:
        scale = max_dim / long_dim
        new_size = (round(image.size[0] * scale), round(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def method_slug(label):
    """Map a method ``LABEL`` to its short token (e.g. "StyTr² transformer …" → "stytr2")."""
    return label.split()[0].replace("²", "2").lower()


def output_filename(slug, content_stem, style_stem, ext="webp"):
    """Output filename convention shared by ``src.app`` (Gradio) and ``src.cli``."""
    return f"{slug}-{content_stem}_X_{style_stem}.{ext}"
