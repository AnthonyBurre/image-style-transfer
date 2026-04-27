import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

def load_image_tensor(image, max_dim):
    """
    Loads a PIL image, fixes orientation/mode, and downsamples it so the longest
    side is at most ``max_dim``, preserving aspect ratio. Returns a batched
    float32 tensor in [0, 1] suitable for the Magenta style transfer model.

    Resampling uses PIL's LANCZOS, which preserves edge sharpness even on large
    downsamples (e.g. 5568→1280, 1920→256) — much better than tf.image.resize's
    default bilinear for this kind of ratio. Images smaller than ``max_dim`` are
    left untouched (no upscaling).
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image).__name__}")

    image = ImageOps.exif_transpose(image).convert("RGB")

    long_dim = max(image.size)
    if long_dim > max_dim:
        scale = max_dim / long_dim
        new_size = (round(image.size[0] * scale), round(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    img = tf.convert_to_tensor(np.array(image), dtype=tf.float32) / 255.0
    return img[tf.newaxis, :]

def tensor_to_image(tensor):
    """
    Converts a float tensor back to a displayable PIL Image.
    
    Args:
        tensor (tf.Tensor): The output tensor from the style transfer model.
        
    Returns:
        PIL.Image: The resulting stylized image.
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    # Remove the batch dimension if it exists
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)
