# Test app for artistic style transfer

A small Gradio web app for image-to-image artistic style transfer. Three methods are available, selectable in the UI per request:

- **Magenta** (`arbitrary-image-stylization-v1-256` from TF Hub) — feed-forward inference, returns in seconds.
- **Gatys VGG19** — optimisation-based neural style transfer (Gatys, Ecker & Bethge, 2015). Slower but produces more abstracted, painterly results.
- **StyTr²** (Deng et al., CVPR 2022) — transformer-based feed-forward inference. ~30 s per image on CPU; tends to preserve content tones (e.g. true blacks) better than the other two. Pretrained weights are pulled from the `datnguyentien204/Sty_TR2_38` mirror on Hugging Face on first use.

## Run with Docker

The simplest, most reproducible option:

```shell
docker build -t style-transfer .
docker run --rm -m "4g" -p 7860:7860 style-transfer
```

Open http://localhost:7860.

Docker Desktop on Mac runs in a Linux VM with no access to the host GPU, so this path is CPU-only. Fine for Magenta; Gatys will take several minutes per image.

## Run on the host

Useful for faster iteration on Gatys, and required to use the GPU on Apple Silicon (see below):

```shell
.venv/bin/python -m src.app
```

### Optional: Metal GPU acceleration (Apple Silicon)

Apple's `tensorflow-metal` plugin lets TensorFlow execute on the M-series GPU via Metal. On an M2, 300-step Gatys drops from several minutes to roughly 30–60 s, and you can raise `MAX_DIM` in `src/gatys_style_transfer.py` to 768 or 1024 to get higher-resolution output.

```shell
.venv/bin/pip install tensorflow-metal
.venv/bin/python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The check command should print a non-empty list containing a GPU device. After that, run the app normally — TensorFlow picks up Metal automatically; no code changes needed.

`tensorflow-metal` is deliberately **not** in `requirements.txt`: it only installs on macOS arm64 and would break the Linux Docker build. Keep it as a host-only install. If `tensorflow-metal` and `tensorflow==2.19.0` ever fall out of sync (the plugin pins to specific TF versions), pip will warn during install — that's almost always the cause if the GPU check returns an empty list afterwards.
