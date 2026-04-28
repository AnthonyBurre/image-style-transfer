import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from .methods import METHODS

_STYLIZE = {m.LABEL: m.stylize for m in METHODS}


def stylize(content_path, style_path, method, progress=gr.Progress()):
    content = Image.open(content_path)
    style = Image.open(style_path)
    result = _STYLIZE[method](content, style, progress=progress)

    method_slug = method.split()[0].replace("²", "2")
    name = f"{method_slug}-{Path(content_path).stem}_X_{Path(style_path).stem}.webp"
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

    # Inputs are filepath (not PIL) so the dispatcher can read the original
    # upload filenames and use their stems to name the downloaded result.
    demo = gr.Interface(
        fn=stylize,
        inputs=[
            gr.Image(type="filepath", label="Content Image"),
            gr.Image(type="filepath", label="Style Image"),
            gr.Radio([m.LABEL for m in METHODS], value=METHODS[0].LABEL, label="Method"),
        ],
        outputs=gr.Image(type="filepath", label="Stylized Result"),
        title="🎨 Artistic Style Transfer",
        description=description,
        flagging_mode="never",
    )

    print("---------------------------------------------------------------------")
    print("If running in a Docker container, access app at: http://localhost:7860")
    print("---------------------------------------------------------------------")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
