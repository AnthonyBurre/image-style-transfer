# Import functions from our other modules
from .model_loader import load_model
from .image_utils import load_image_tensor, tensor_to_image

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
    # Preprocess the images to be in the correct format for the model
    content_tensor = load_image_tensor(content_image)
    style_tensor = load_image_tensor(style_image)

    print("Applying style transfer...")
    # Apply the style transfer model
    stylized_image_tensor = hub_model(content_tensor, style_tensor)[0]

    print("Conversion complete. Returning final image.")
    # Convert the output tensor back to a displayable PIL image
    final_image = tensor_to_image(stylized_image_tensor)

    return final_image
