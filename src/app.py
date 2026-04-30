import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from .methods import METHODS

_STYLIZE = {m.LABEL: m.stylize for m in METHODS}
_REQUIRES_SQUARE = {m.LABEL: m.REQUIRES_SQUARE for m in METHODS}

PREVIEW_HEIGHT = 320


def _editor_composite(editor_value):
    if not editor_value:
        return None
    return editor_value.get("composite")


def stylize(basic_path, editor_value, style_path, method, progress=gr.Progress()):
    content_path = _editor_composite(editor_value) if _REQUIRES_SQUARE[method] else basic_path
    if content_path is None or style_path is None:
        return None

    content = Image.open(content_path)
    style = Image.open(style_path)
    result = _STYLIZE[method](content, style, progress=progress)

    method_slug = method.split()[0].replace("²", "2")
    name = f"{method_slug}-{Path(content_path).stem}_X_{Path(style_path).stem}.webp"
    out_path = Path(tempfile.mkdtemp()) / name
    result.save(out_path, format="webp", lossless=True)
    return str(out_path)


def _on_method_change(method, basic_path, editor_value):
    """Toggle the basic-image / editor pair when the active method's crop
    requirement changes. Whichever component was holding the user's content
    image hands its value to the other so the upload survives the swap.
    """
    if _REQUIRES_SQUARE[method]:
        return (
            gr.update(visible=False),
            gr.update(value=basic_path, visible=True),
        )
    return (
        gr.update(value=_editor_composite(editor_value), visible=True),
        gr.update(visible=False),
    )


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
                    visible=not METHODS[0].REQUIRES_SQUARE,
                )
                # Editor is the stacked alternative, shown only for methods that
                # need a square input. transforms=("crop",) plus brush/eraser/
                # layers off leaves just the crop tool.
                content_editor = gr.ImageEditor(
                    type="filepath",
                    image_mode="RGB",
                    label="Content Image — drag to choose square crop",
                    height=PREVIEW_HEIGHT,
                    transforms=("crop",),
                    brush=False,
                    eraser=False,
                    layers=False,
                    sources=("upload", "clipboard"),
                    visible=METHODS[0].REQUIRES_SQUARE,
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

        method_radio.change(
            _on_method_change,
            inputs=[method_radio, content_basic, content_editor],
            outputs=[content_basic, content_editor],
        )
        run_button.click(
            stylize,
            inputs=[content_basic, content_editor, style_image, method_radio],
            outputs=output_image,
        )

    print("---------------------------------------------------------------------")
    print("If running in a Docker container, access app at: http://localhost:7860")
    print("---------------------------------------------------------------------")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
