import gradio as gr

from .style_transfer import perform_style_transfer
from .gatys_style_transfer import perform_gatys_style_transfer
from .stytr2_style_transfer import perform_stytr2_style_transfer

MAGENTA = "Magenta (fast — ~seconds)"
GATYS = "Gatys VGG19 (slow — minutes, more abstracted)"
STYTR2 = "StyTr² transformer (medium — ~30s, sharper detail)"


def stylize(content_image, style_image, method, progress=gr.Progress()):
    if method == GATYS:
        return perform_gatys_style_transfer(content_image, style_image, progress=progress)
    if method == STYTR2:
        return perform_stytr2_style_transfer(content_image, style_image)
    return perform_style_transfer(content_image, style_image)


def main():
    """Defines and launches the Gradio web interface."""
    demo = gr.Interface(
        fn=stylize,
        inputs=[
            gr.Image(type="pil", label="Content Image"),
            gr.Image(type="pil", label="Style Image"),
            gr.Radio([MAGENTA, GATYS, STYTR2], value=MAGENTA, label="Method"),
        ],
        outputs=gr.Image(type="pil", label="Stylized Result"),
        title="🎨 Artistic Style Transfer PoC",
        description="""
        This is a Proof of Concept for Artistic Style Transfer.
        Upload a **Content Image** (the picture you want to transform) and a **Style Image** (the artwork whose style you want to copy).
        Pick a method — **Magenta** is a feed-forward model that returns in seconds; **Gatys VGG19** runs an optimisation loop (slow, but produces more abstracted/painterly stylisation); **StyTr²** is a transformer-based feed-forward model that tends to preserve content tones (incl. true blacks) better than the other two.
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
