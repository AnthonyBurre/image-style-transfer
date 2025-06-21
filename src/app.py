import gradio as gr
from PIL import Image
import os

# Import the core function from our style_transfer module
from .style_transfer import perform_style_transfer

def create_examples():
    """Creates placeholder example images if they don't exist."""
    print("Creating example files...")
    try:
        os.makedirs("examples", exist_ok=True)
        
        files_to_check = {
            "content.jpg": Image.new('RGB', (400, 300), color='blue'),
            "style.jpg": Image.new('RGB', (300, 400), color='red'),
            "chicago.jpg": Image.new('RGB', (400, 300), color='grey'),
            "van_gogh.jpg": Image.new('RGB', (300, 400), color='yellow')
        }
        
        for filename, img in files_to_check.items():
            path = os.path.join("examples", filename)
            if not os.path.exists(path):
                img.save(path)
                
        print("Example files are ready. You can replace them with your own images.")
    except Exception as e:
        print(f"Could not create example files (this is okay): {e}")

def main():
    """Defines and launches the Gradio web interface."""
    
    # First, make sure our examples are available for the UI
    create_examples()

    # Define the Gradio interface
    demo = gr.Interface(
        fn=perform_style_transfer,
        inputs=[
            gr.Image(type="pil", label="Content Image"),
            gr.Image(type="pil", label="Style Image")
        ],
        outputs=gr.Image(type="pil", label="Stylized Result"),
        title="🎨 Artistic Style Transfer PoC",
        description="""
        This is a Proof of Concept for Artistic Style Transfer.
        Upload a **Content Image** (the picture you want to transform) and a **Style Image** (the artwork whose style you want to copy).
        The model will blend the two to create a new piece of art!
        """,
        allow_flagging="never",
        examples=[
            ["examples/content.jpg", "examples/style.jpg"],
            ["examples/chicago.jpg", "examples/van_gogh.jpg"]
        ]
    )
    
    # Launch the web interface
    print("Launching Gradio interface... Open the URL in your browser.")
    demo.launch()

# This makes the script runnable
if __name__ == "__main__":
    main()
