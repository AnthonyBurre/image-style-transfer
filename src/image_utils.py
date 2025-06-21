import tensorflow as tf
import numpy as np
from PIL import Image

def load_image_tensor(image, max_dim=512):
    """
    Loads a PIL image, converts it to a float tensor, and resizes it
    while maintaining its aspect ratio.
    
    Args:
        image (PIL.Image): The input image.
        max_dim (int): The maximum dimension (width or height) of the resized image.
        
    Returns:
        tf.Tensor: The processed image tensor, ready for the model.
    """
    # Convert PIL Image to a TensorFlow tensor
    img = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    
    # Get the shape *before* adding the batch dimension. It's (height, width, channels)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)  # shape is [height, width]

    # Find the longest dimension
    long_dim = tf.reduce_max(shape)

    # Calculate the scaling factor
    scale = max_dim / long_dim

    # Calculate the new shape, which will be a 2-element tensor [new_height, new_width]
    new_shape = tf.cast(shape * scale, tf.int32)

    # Add the batch dimension for the model
    img = img[tf.newaxis, :]
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Resize the batch of images
    img = tf.image.resize(img, new_shape)

    return img

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
