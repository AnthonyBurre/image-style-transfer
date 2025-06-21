import os
import requests
import tarfile
import tensorflow_hub as hub

def download_and_extract_model(model_url, model_dir="model"):
    """
    Downloads and extracts the model if it doesn't already exist.
    This is a robust way to handle potential SSL certificate issues.
    
    Args:
        model_url (str): The URL to download the compressed model from.
        model_dir (str): The local directory to store the model.

    Returns:
        str: The path to the local model directory.
    """
    if os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' already exists. Skipping download.")
        return model_dir

    print(f"Downloading model from {model_url}...")
    os.makedirs(model_dir, exist_ok=True)
    
    # Use requests to download the file, as it's often more reliable with SSL
    response = requests.get(model_url, stream=True)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

    file_path = os.path.join(model_dir, "model.tar.gz")
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("Download complete. Extracting model...")
    with tarfile.open(file_path) as tar:
        tar.extractall(path=model_dir)
        
    os.remove(file_path)  # Clean up the downloaded archive
    print("Extraction complete.")
    return model_dir

def load_model():
    """
    Ensures the model is downloaded and loads it into memory.
    
    Returns:
        A loaded TensorFlow Hub model object.
    """
    # Define the URL for the compressed model from TF Hub
    model_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed'
    local_model_path = download_and_extract_model(model_url)
    
    print(f"Loading Style Transfer model from local path: {local_model_path}")
    hub_model = hub.load(local_model_path)
    print("Model loaded successfully.")
    return hub_model

