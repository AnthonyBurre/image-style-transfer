# Artistic Style Transfer Investigation

Remixing the content of one image into the style of another is the type of task that invites a variety of creative mathematical approaches. This project serves as a comparison of some of these methodologies, and a tool for testing them out.

- **Magenta** (`arbitrary-image-stylization-v1-256` from TF Hub) - feed-forward inference, returns in seconds.
- **Gatys VGG19** - optimisation-based neural style transfer (Gatys, Ecker & Bethge, 2015). Slower but produces more abstracted, painterly results.
- **StyTr²** (Deng et al., CVPR 2022) - transformer-based feed-forward inference. ~30 s per image on CPU; tends to preserve content tones (e.g. true blacks) better than the other two. Pretrained weights are pulled from the `datnguyentien204/Sty_TR2_38` mirror on Hugging Face on first use.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then sync the extra that matches your hardware:

| Hardware                              | Sync command                              |
| ------------------------------------- | ----------------------------------------- |
| Any CPU (Mac Intel, Linux, Windows)   | `uv sync --extra cpu`                     |
| Apple Silicon (Metal GPU)             | `uv sync --extra cpu --extra metal`       |
| NVIDIA GPU (Linux native, or WSL2)    | `uv sync --extra cuda`                    |

AMD/Intel GPUs have no working path.

Run the GUI at http://localhost:7860 for interactive exploration:

```shell
uv run python -m src.app
```

Use the CLI for scripting/batch processing:

```shell
uv run python -m src.cli \
  -c examples/content/lighthouse.png \
  -s examples/style/ty.png \
  -o out.png -m magenta
```
See all flags:

```shell
uv run python -m src.cli -h
```

The CLI accepts directories on `-c` / `-s` for batch runs — see `-h` for all flags.

## Docker option

CPU-only, but no need to install uv or python:

```shell
docker build -t style-transfer .
docker run --rm -m 4g -p 7860:7860 style-transfer
```

CLI variant, with a host-mounted working dir so the output file lands back on the host:

```shell
docker run --rm -m 4g \
  -v "$PWD:/work" -w /work \
  -v "$PWD/model-stytr2:/app/model-stytr2" \
  style-transfer \
  src.cli -c examples/content/hoodwinked.png \
          -s examples/style/spiderverse.png \
          -o out.png -m stytr2
```

On Windows PowerShell replace `$PWD` with `${PWD}`; in `cmd.exe` use `%cd%`. Mount `$PWD/model:/app/model` instead when running Magenta, otherwise its weights re-download every invocation.

## Examples

For those of you who are too busy to clone and run this yourself, I've included some examples here. A good model will work its magic on any sort of content image respectably, but it is more up to the user to select an optimal style image for best results.

### Content images (`examples/content/`)

<table>
<tr>
<td align="center" width="33%"><img src="examples/content/katy.png" width="100%"><br><b>Katy</b></td>
<td align="center" width="33%"><img src="examples/content/lighthouse.png" width="100%"><br><b>Lighthouse</b></td>
<td align="center" width="33%"><img src="examples/content/hoodwinked.png" width="100%"><br><b>Hoodwinked!</b></td>
</tr>
<tr>
<td align="center"><sub>640×640. Dog portrait with dense fur texture and an off-centre face. Square aspect ratio, so StyTr² operates on the full frame with no crop.</sub></td>
<td align="center"><sub>480×640. White architectural form against a wispy-cloud sky - hard high-contrast edges beside broad smooth gradient regions, the "kind" content where every method produces something defensible.</sub></td>
<td align="center"><sub>912×513. A CGI animation still - content that is already heavily stylized in its source rendering, so transfer is layered on top of an existing look rather than a photograph. Wide aspect, so StyTr² centre-crops to square.</sub></td>
</tr>
</table>

### Style images (`examples/style/`)

<table>
<tr>
<td align="center" width="33%"><img src="examples/style/ty.png" width="100%"><br><b>Painted desert landscape</b><br><sub><i>ty.png</i></sub></td>
<td align="center" width="33%"><img src="examples/style/edgerunners.png" width="100%"><br><b><i>Cyberpunk: Edgerunners</i></b><br><sub>2022</sub></td>
<td align="center" width="33%"><img src="examples/style/spiderverse.png" width="100%"><br><b><i>Spider-Man: Into the Spider-Verse</i></b><br><sub>2018</sub></td>
</tr>
<tr>
<td align="center"><sub>Visible brushwork and a saturated landscape palette - the closest analogue here to the painted style images the original NST papers used; best showcase for <b>Gatys</b>'s painterly Gram-matrix abstraction.</sub></td>
<td align="center"><sub>Anime aesthetic: flat colour regions, bold line work, neon highlights against deep shadow. Modern digital-animation style rather than a traditional painting - tests how the feed-forward methods cope with a non-painterly style code.</sub></td>
<td align="center"><sub>Halftone dots, chromatic aberration, comic-book outlines and a vibrant complementary palette. The high-frequency dot/line pattern is the kind of mark Magenta's global style code tends to dissolve and StyTr²'s patch-attention preserves better.</sub></td>
</tr>
</table>

### Sizing and aspect ratio

The example dimensions are deliberately non-uniform: aspect ratio is preserved on disk because each method handles non-square inputs differently.

- **Magenta** runs the transfer network fully-convolutionally, so output resolution tracks the content image (up to a 2048 px longest-side cap).
- **Gatys** downsamples both inputs to 512 px longest-side (a CPU/4 GB-container budget cap).
- **StyTr²** does a square center-crop to 512×512 - the transformer's patch-grid reshape requires `H == W`. Wide content like *Hoodwinked!* loses its outer regions when run through StyTr². This is a real behaviour of the model, not a bug, and the example images are sized so it is visible.



## Models

The three methods are one representative from each of the three dominant architectural buckets of dedicated neural style transfer, in chronological order:

1. **Optimization-based** (Gatys, 2015) - the image itself is the parameter; no learned mapping.
2. **Feed-forward CNN** (Magenta, 2017) - learned encoder/decoder, single global style code via conditional instance norm.
3. **Feed-forward transformer** (StyTr², 2022) - patch tokens with content↔style cross-attention.

The trio also brackets the speed/quality tradeoff: minutes per image for Gatys's painterly optimization, seconds for Magenta's averaged feed-forward, ~30 s for StyTr² in between.

The pointed omission is **diffusion-prior style transfer** (ControlNet, IP-Adapter, InstantStyle, StyleAligned, B-LoRA, …) - arguably the dominant paradigm as of 2026, where the field stopped training dedicated style-transfer networks and started conditioning generic text-to-image diffusion models instead. So: a comprehensive demonstration of how the problem was approached while style transfer was still its own architecture, and a deliberately incomplete demonstration of how it's approached today. Also out of scope: pre-neural patch-based methods (Image Quilting, Efros & Freeman 2001), GAN domain transfer (CycleGAN and descendants), and video/3D/NeRF variants.

### Magenta - feed-forward arbitrary stylization

Ghiasi et al., Google Magenta (2017). A style-prediction sub-network compresses the style image into a low-dimensional embedding that parametrizes conditional instance-norm layers in a transfer network operating on the content image; the stylized image comes out in a single forward pass - no per-image optimization, no attention. That is why it returns in seconds, and also why it tends toward a smoother, more "averaged" stylization than the other two: a single global style vector cannot localize fine ornament or hard edges in the style image to specific regions of the content. Colour palettes transfer well; high-frequency style detail (Spider-Verse's halftone dots, Edgerunners' line work) tends to dissolve into the content's textures rather than persist as discrete marks.

<!-- example outputs go here -->

### Gatys VGG19 - optimization-based neural style transfer

The original Gatys, Ecker & Bethge (2015) formulation. Treats style transfer as an inverse problem: initialize from the content image, then minimize a weighted sum of a content loss (MSE on `block5_conv2` activations), a style loss (MSE on Gram matrices across `block{1..5}_conv1`), and a total-variation regularizer, via Adam over the pixel tensor for ~300 steps per request. There is no learned style mapping - the model parameters are the output pixels themselves, which is what makes it slow. Because Gram matrices encode texture statistics rather than spatial layout, output is markedly more abstracted and painterly than the feed-forward methods: content geometry survives, but objects bleed into the style's brushwork and palette in a way the others never quite manage. The painted-landscape (*ty.png*) column of the example matrix is where this is most legible.

Notes:
You can raise `MAX_DIM` in `src/methods/gatys.py` to 768 or 1024 for higher-resolution output once Metal is in play.

<!-- example outputs go here -->

### StyTr² - transformer-based arbitrary style transfer

Deng et al. (CVPR 2022). A pure-transformer alternative to both CNN feed-forward (Magenta) and per-image optimization (Gatys). Content and style images are tokenized via a stride-8 patch embedding, encoded by separate transformer stacks with content-aware positional encoding (CAPE), and fused by a cross-attention decoder before a convolutional upsampler returns to image space. Because attention operates patch-wise rather than through a single global style code, fine style detail and content tonality (notably true blacks) survive better than in Magenta, while inference stays feed-forward - runtime sits between the other two at ~30 s on CPU at 512², bounded by the O(N²) attention over 64×64 = 4096 tokens. The price: the patch-grid reshape requires `H == W`, so wide content is centre-cropped to a square before inference. Pretrained weights are pulled from the `datnguyentien204/Sty_TR2_38` Hugging Face mirror on first use.

Notes:  
Apple silicon:  StyTr² stays on CPU here — PyTorch's MPS backend hits [pytorch#96056](https://github.com/pytorch/pytorch/issues/96056) on this model's adaptive-pool op.
StyTr² prints the device it loaded onto on first call.


<!-- example outputs go here -->
