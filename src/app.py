import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from .image import output_filename
from .methods import METHODS

_STYLIZE = {m.LABEL: m.stylize for m in METHODS}

PREVIEW_HEIGHT = 320


def stylize(content_path, style_path, method, progress=gr.Progress()):
    if content_path is None or style_path is None:
        return None

    content = Image.open(content_path)
    style = Image.open(style_path)
    result = _STYLIZE[method](content, style, progress=progress)

    method_slug = method.split()[0].replace("²", "2").lower()
    name = output_filename(method_slug, Path(content_path).stem, Path(style_path).stem)
    out_path = Path(tempfile.mkdtemp()) / name
    result.save(out_path, format="webp", lossless=True)
    return str(out_path)


def main():
    """Defines and launches the Gradio web interface."""
    blurbs = "; ".join(m.BLURB for m in METHODS)
    description = (
        "This is an exploration of pre-diffusion Image Style Transfer.\n"
        "Upload a **Content Image** (the picture you want to transform) and a "
        "**Style Image** (the artwork whose style you want to copy).\n"
        f"Pick a method — {blurbs}."
    )

    with gr.Blocks(title="🎨 Artistic Style Transfer") as demo:
        gr.Markdown(f"# 🎨 Artistic Style Transfer\n{description}")

        with gr.Row():
            with gr.Column(scale=1):
                content_basic = gr.Image(
                    type="filepath",
                    label="Content Image",
                    height=PREVIEW_HEIGHT,
                )
            with gr.Column(scale=1):
                style_image = gr.Image(
                    type="filepath",
                    label="Style Image",
                    height=PREVIEW_HEIGHT,
                )

        with gr.Row():
            with gr.Column(scale=1):
                method_radio = gr.Radio(
                    [m.LABEL for m in METHODS],
                    value=METHODS[0].LABEL,
                    label="Method",
                )
                run_button = gr.Button("Stylize", variant="primary")
            with gr.Column(scale=1):
                output_image = gr.Image(
                    type="filepath",
                    label="Stylized Result",
                    height=PREVIEW_HEIGHT,
                    interactive=False,
                )

        run_button.click(
            stylize,
            inputs=[content_basic, style_image, method_radio],
            outputs=output_image,
        )

    print("---------------------------------------------------------------------")
    print("If running in a Docker container, access app at: http://localhost:7860")
    print("---------------------------------------------------------------------")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
