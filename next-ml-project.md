# Setting up the next ML project, GPU-friendly

A reference for initializing a cross-platform ML project so GPU support is honest from day one — instead of bolting it on later and writing "CPU-only by design" caveats.

## The shape of the problem

A single Docker image can't cover Mac + Linux + Windows users *and* offer GPU support. GPU-in-Docker is Linux-host-only: Docker Desktop on macOS doesn't pass GPUs through, and Apple Silicon's Metal can't be exposed to a container at all. So if cross-platform matters, host-install has to be a first-class path — and Docker is optional convenience on top.

## Recommended stack

**Environment manager:** [uv](https://docs.astral.sh/uv/). Fast, deterministic lockfile, and handles the CPU-vs-GPU torch index split cleanly via dependency groups. (Or [pixi](https://pixi.sh/) if you need conda packages.)

**`pyproject.toml` sketch** — one project, two extras, two PyTorch indexes:

```toml
[project]
name = "your-project"
requires-python = ">=3.10"
dependencies = [
    "gradio",
    "pillow",
    # ...non-torch deps
]

[project.optional-dependencies]
cpu = ["torch", "torchvision"]
cuda = ["torch", "torchvision"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cuda" }]]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cuda", extra = "cuda" },
]
torchvision = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cuda", extra = "cuda" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

**Sync commands users run:**

```shell
uv sync --extra cpu     # laptops, Mac (MPS picked up at runtime), Windows CPU
uv sync --extra cuda    # Linux + NVIDIA, or WSL2
```

The lockfile pins both resolutions, so anyone on the project can switch extras without drift.

## Docker: when yes, when no

- **Skip Docker** if your audience is comfortable with `uv sync`. One less thing to maintain.
- **Small CPU-only image** if you want a "no Python install required" demo path. Keep it `python:3.x-slim` + the `cpu` extra. ~1 GB.
- **Separate `Dockerfile.cuda`** on `nvidia/cuda:X.Y-cudnn-runtime` only if you have a real Linux-server deployment target. Don't try to make one image do both.

Don't pretend a single image covers everyone — that's the trap the previous project fell into.

## Hosted demo path

If the project is web-shaped (Gradio, Streamlit, FastAPI), publish the public demo on [Hugging Face Spaces](https://huggingface.co/spaces) or [Modal](https://modal.com/). Both give you free GPU minutes and handle the container + CUDA wheel matching for you. Local repo stays focused on the dev experience; the demo lives where the GPU lives.

## Device handling in code

For PyTorch, pick the device once at load time:

```python
def resolve_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

Then `.to(device)` the model and the inputs. **Print the resolved device** on first use — users debugging "why is this slow" need to see which path they're on without adding instrumentation. MPS support varies by op; test it before promising it in your README.

For TensorFlow, device pickup is implicit on macOS (with `tensorflow-metal`) and Linux (with a CUDA-built TF). Document the install step; don't try to do device selection in code.

## The tradeoff

More upfront environment plumbing — lockfile, conflicting extras, possibly two Dockerfiles — in exchange for honest GPU support across platforms. Worth it whenever you expect users to install themselves.
