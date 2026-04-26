import gradio as gr

from .style_transfer import perform_style_transfer

def main():
    """Defines and launches the Gradio web interface."""
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
        flagging_mode="never",
    )

    print("---------------------------------------------------------------------")
    print(f"If running in a Docker container, access app at: http://localhost:7860")
    print("---------------------------------------------------------------------")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    main()
