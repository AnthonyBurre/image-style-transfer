# Import functions from our other modules
from .model_loader import load_model
from .image_utils import load_image_tensor, tensor_to_image

# The Magenta arbitrary-image-stylization-v1-256 model was trained with the style
# image at 256px. Feeding it a larger style image is what causes the "patches of
# the style image pasted onto the content" failure mode — at native scale the
# network treats style features as content. Content is capped at 1280 to stay
# inside the container's 4 GB memory budget for very large inputs (e.g. 5568px).
CONTENT_MAX_DIM = 1280
STYLE_MAX_DIM = 256

# Load the model once when this module is imported
hub_model = load_model()

def perform_style_transfer(content_image, style_image):
    """
    Takes content and style images, applies style transfer, and returns the result.
    This is the core function that orchestrates the style transfer process.

    Args:
        content_image (PIL.Image): The image providing the content.
        style_image (PIL.Image): The image providing the artistic style.

    Returns:
        PIL.Image: The final stylized image.
    """
    if content_image is None or style_image is None:
        return None  # Return nothing if inputs are missing

    print("Processing images...")
    content_tensor = load_image_tensor(content_image, max_dim=CONTENT_MAX_DIM)
    style_tensor = load_image_tensor(style_image, max_dim=STYLE_MAX_DIM)

    print("Applying style transfer...")
    stylized_image_tensor = hub_model(content_tensor, style_tensor)[0]

    print("Conversion complete. Returning final image.")
    return tensor_to_image(stylized_image_tensor)
