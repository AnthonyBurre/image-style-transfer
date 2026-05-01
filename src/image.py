"""PIL helpers shared by method modules.

Each method then wraps the returned PIL image into its own framework tensor
(TF for Magenta/Gatys, torch for StyTr²).
"""
from PIL import Image, ImageOps


def prepare(image, max_dim):
    """EXIF-corrected RGB PIL image, longest side ≤ ``max_dim``.

    LANCZOS resampling preserves edges on large downsamples (e.g. 5568→1280)
    much better than bilinear. Smaller images are not upscaled.
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


def output_filename(method_slug, content_stem, style_stem):
    """Output filename convention shared by ``src.app`` (Gradio) and ``src.cli``."""
    return f"{method_slug}-{content_stem}_X_{style_stem}.webp"
