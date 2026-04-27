import gradio as gr

from .methods import METHODS

_STYLIZE = {m.LABEL: m.stylize for m in METHODS}


def stylize(content_image, style_image, method, progress=gr.Progress()):
    return _STYLIZE[method](content_image, style_image, progress=progress)


def main():
    """Defines and launches the Gradio web interface."""
    blurbs = "; ".join(m.BLURB for m in METHODS)
    description = (
        "This is a Proof of Concept for Artistic Style Transfer.\n"
        "Upload a **Content Image** (the picture you want to transform) and a "
        "**Style Image** (the artwork whose style you want to copy).\n"
        f"Pick a method — {blurbs}."
    )

    demo = gr.Interface(
        fn=stylize,
        inputs=[
            gr.Image(type="pil", label="Content Image"),
            gr.Image(type="pil", label="Style Image"),
            gr.Radio([m.LABEL for m in METHODS], value=METHODS[0].LABEL, label="Method"),
        ],
        outputs=gr.Image(type="pil", label="Stylized Result"),
        title="🎨 Artistic Style Transfer PoC",
        description=description,
        flagging_mode="never",
    )

    print("---------------------------------------------------------------------")
    print("If running in a Docker container, access app at: http://localhost:7860")
    print("---------------------------------------------------------------------")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
