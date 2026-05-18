"""Headless CLI: stylise (content, style) image pairs and write the result(s).

The bare invocation ``python -m src.cli`` batch-processes every image in
``examples/content`` against every image in ``examples/style`` through the
Magenta model, writing ``.webp`` files into ``examples/output/`` using the
same filename convention as the Gradio app.

Each of ``-c`` / ``-s`` may be a file or a directory; with directory inputs
the cartesian product of (content, style) pairs is processed. ``-o`` is
either an output file (single-pair only) or an output directory.

Sibling of ``src.app`` (the Gradio UI); the two share method modules but
not dispatch code, so UI changes can't ripple into the CLI.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

from .image import output_filename
from .methods import METHODS

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _slug(label):
    """Map a method ``LABEL`` to its CLI token (e.g. "StyTr² transformer …" → "stytr2")."""
    return label.split()[0].replace("²", "2").lower()


_BY_SLUG = {_slug(m.LABEL): m for m in METHODS}


def _resolve_inputs(path):
    """Return a sorted list of image paths for a file-or-directory argument."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in _IMAGE_EXTS)
        if not files:
            sys.exit(f"error: no image files in {p}")
        return files
    sys.exit(f"error: path not found: {p}")


def run(content_arg, style_arg, output_arg, slug, ext):
    contents = _resolve_inputs(content_arg)
    styles = _resolve_inputs(style_arg)
    method = _BY_SLUG[slug]

    out = Path(output_arg)
    is_file_output = out.suffix.lower() in _IMAGE_EXTS
    total = len(contents) * len(styles)

    if is_file_output and total > 1:
        sys.exit(
            f"error: file output ({out}) requires exactly one content × one style; "
            f"got {len(contents)} × {len(styles)}"
        )

    if not is_file_output:
        out.mkdir(parents=True, exist_ok=True)

    i = 0
    for content_path in contents:
        content = Image.open(content_path)
        for style_path in styles:
            i += 1
            sys.stderr.write(f"[{i}/{total}] {content_path.stem} × {style_path.stem}\n")
            sys.stderr.flush()

            style = Image.open(style_path)
            result = method.stylize(content, style, progress=None)

            out_path = out if is_file_output else out / output_filename(
                slug, content_path.stem, style_path.stem, ext
            )
            result.save(out_path)
            print(str(out_path.resolve()))


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Stylise (content, style) image pairs and write the results. "
                    "Each of -c/-s may be a single image or a directory of images; "
                    "with directory inputs every (content, style) pair is processed. "
                    "For the web UI, run `python -m src.app` instead.",
    )
    parser.add_argument(
        "-c", "--content", default="examples/content",
        help="content image, or directory of content images (default: examples/content)",
    )
    parser.add_argument(
        "-s", "--style", default="examples/style",
        help="style image, or directory of style images (default: examples/style)",
    )
    parser.add_argument(
        "-o", "--output", default="examples/output",
        help="output image path (.png/.jpg/.webp/…) for single-pair, "
             "or output directory for batch (default: examples/output)",
    )
    parser.add_argument(
        "-m", "--method", default="magenta", choices=list(_BY_SLUG),
        help="stylisation method (default: magenta)",
    )
    parser.add_argument(
        "-e", "--ext", default="webp",
        choices=["webp", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="output file extension for batch runs (default: webp). "
             "Ignored when -o is a single file path — the extension on -o wins there.",
    )
    args = parser.parse_args()

    run(args.content, args.style, args.output, args.method, args.ext)


if __name__ == "__main__":
    main()
