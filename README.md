# Artistic Style Transfer Investigation

Remixing the content of one image into the style of another is the type of task that invites a variety of creative mathematical approaches. This project serves as a comparison of some of these methodologies, and a tool for testing them out.

1. **Patch-based** (Image Analogies, Hertzmann et al., 2001) - the pre-neural baseline. No learning and no features beyond raw pixel statistics: for each output pixel, search the style image for a 5×5 luminance patch whose neighbourhood best matches the content image around that point, then copy the corresponding style pixel into the output. Slow and visibly softer than the neural methods - included as the historical bookend.
2. **Optimization-based** (Gatys, Ecker & Bethge, 2015) - the founding approach, and the one that established neural style transfer as a problem at all. Each output image is iteratively optimized from scratch against content and style targets. Slow, but produces strikingly painterly results.
3. **Feed-forward CNN** (Magenta, 2017) - Google's answer to the speed problem. A learned network emits a stylized image in a single forward pass; results return in seconds, at the cost of a smoother, more averaged stylization than Gatys produces.
4. **Transformer** (StyTr², 2022) - feed-forward like Magenta, but trades convolutional encoders for attention over image patches, addressing Magenta's tendency to lose fine style detail and content tonality.

The pointed omission is **diffusion** based style transfer, explored in the [sibling project](https://github.com/AnthonyBurre/diffusion-style-transfer).

## Quick start

With [uv](https://docs.astral.sh/uv/) installed, sync the extra that matches your hardware:

| Hardware                              | Sync command                              |
| ------------------------------------- | ----------------------------------------- |
| Any CPU*                              | `uv sync --extra cpu`                     |
| Apple Silicon (Metal GPU)             | `uv sync --extra cpu --extra metal`       |
| NVIDIA GPU (Linux native, or WSL2)    | `uv sync --extra cuda`                    |

<sub>*AMD/Intel GPUs have no working path.</sub>

Run the GUI at http://localhost:7860 for interactive exploration:

```shell
uv run python -m src.app
```

Use the CLI for scripting/batch processing:

```shell
uv run python -m src.cli \
  -c examples/content/lighthouse.png \
  -s examples/style/ty.png \
  -o out.png -m gatys
```
Accepts directories on `-c` / `-s` for batch runs, see `-h` for all flags:

```shell
uv run python -m src.cli -h
```

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

## Example Images

For those of you who are too busy to clone and run this yourself, I've included some examples here. A good model will work its magic on any sort of content image respectably, but it is more up to the user to select an optimal style image for best results.

### `examples/content/`

<table>
<tr>
<td align="center" width="33%"><img src="examples/content/katy.png" width="100%"></td>
<td align="center" width="33%"><img src="examples/content/hoodwinked.png" width="100%"></td>
<td align="center" width="33%"><img src="examples/content/lighthouse.png" width="100%"></td>
</tr>
<tr>
<td align="center"><sub>640x640. My absolutely perfect dog Katy taking a rest during a sunny walk on some nice grass.</sub></td>
<td align="center"><sub>912×513. A frame of an old film whose graphics could use an update. Is style transfer the answer?</sub></td>
<td align="center"><sub>535×640. White architectural form against a wispy-cloud sky - hard high-contrast edges beside broad smooth gradient regions.</sub></td>
</tr>
</table>

### `examples/style/`

<table>
<tr>
<td align="center" width="33%"><img src="examples/style/ty.png" width="100%"></td>
<td align="center" width="33%"><img src="examples/style/kandinsky.png" width="100%"></td>
<td align="center" width="33%"><img src="examples/style/spiderverse.png" width="100%"></td>
</tr>
<tr>
<td align="center"><sub>My little brother's painting - Visible brushwork and a landscape palette - loose, painterly marks rather than hard graphic edges.</sub></td>
<td align="center"><sub>Kandinsky's <i>Improvisation 28</i> - early abstract expressionism. Likely in the training datasets the original NST papers used.</sub></td>
<td align="center"><sub>Frame from Spiderverse movie - Halftone dots, chromatic aberration, comic-book outlines and a vibrant complementary palette.</sub></td>
</tr>
</table>

## Models

This project compares four methodologies of image style transfer, spanning the non-neural patch baseline of 2001 through contemporary transformer-based attention mechanisms. All the models here take exactly one content and one style image as input, but it would be interesting to look into models like CycleGAN (which would take a collection of style images) in the future. Arbitrary-style GAN variants (AdaIN-WCT, MUNIT) do accept a single style image at inference, but are mechanically a flavour of feed-forward CNN and don't open a new era beyond what Magenta already represents here.


### Image Analogies - patch-based pre-neural baseline

Hertzmann, Jacobs, Oliver, Curless & Salesin (SIGGRAPH 2001). The pre-neural bookend, included to show what style transfer looked like before deep learning entered the picture. No learning and no features beyond raw pixel statistics: for each output pixel, find the 5×5 luminance patch in the style image whose neighbourhood best matches the content image around that point, then copy the corresponding style pixel into the output. A 4-level Gaussian pyramid threads coarse structure into fine synthesis, and an Ashikhmin coherence term prefers source coordinates "shifted" from already-synthesised neighbours over isolated nearest-neighbour matches - without it the output is patchy noise; with it, marks and brushwork survive across neighbouring pixels. Matching happens on YIQ luminance only so structure (not colour) drives the search, with the style image's IQ chroma channels transferred unchanged to give the output the style's palette.

Visibly softer and noisier than the neural methods, which is the point - the patchy character is what makes the contrast with everything Gatys 2015 onward legible.

Notes:
Both inputs cap at 512 px longest-side. Runtime ~1 min for rectangular inputs, up to ~6 min for full 512² square content on CPU. No GPU path; pre-neural is pre-GPU.

<table>
<tr>
<td></td>
<td align="center" width="25%"><img src="examples/content/katy.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/hoodwinked.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/lighthouse.png" width="100%"></td>
</tr>
<tr>
<td align="center" width="25%"><img src="examples/style/ty.png" width="100%"></td>
<td align="center"><img src="examples/output/analogies-katy_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/analogies-hoodwinked_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/analogies-lighthouse_X_ty.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/kandinsky.png" width="100%"></td>
<td align="center"><img src="examples/output/analogies-katy_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/analogies-hoodwinked_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/analogies-lighthouse_X_kandinsky.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/spiderverse.png" width="100%"></td>
<td align="center"><img src="examples/output/analogies-katy_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/analogies-hoodwinked_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/analogies-lighthouse_X_spiderverse.webp" width="100%"></td>
</tr>
</table>

### Gatys VGG19 - optimization-based neural style transfer

The original Gatys, Ecker & Bethge (2015) formulation. Treats style transfer as an inverse problem: initialize from the content image, then minimize a weighted sum of a content loss (MSE on `block5_conv2` activations), a style loss (MSE on Gram matrices across `block{1..5}_conv1`), and a total-variation regularizer, via Adam over the pixel tensor for ~300 steps per request. There is no learned style mapping - the model parameters are the output pixels themselves, which is what makes it slow. Because Gram matrices encode texture statistics rather than spatial layout, output is markedly more abstracted and painterly than the feed-forward methods: content geometry survives, but objects bleed into the style's brushwork and palette in a way the others never quite manage.

Notes:
Both inputs are downsampled to 512 px longest-side. Raise `MAX_DIM` in `src/methods/gatys.py` to 768 or 1024 for higher-resolution output.

<table>
<tr>
<td></td>
<td align="center" width="25%"><img src="examples/content/katy.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/hoodwinked.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/lighthouse.png" width="100%"></td>
</tr>
<tr>
<td align="center" width="25%"><img src="examples/style/ty.png" width="100%"></td>
<td align="center"><img src="examples/output/gatys-katy_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/gatys-hoodwinked_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/gatys-lighthouse_X_ty.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/kandinsky.png" width="100%"></td>
<td align="center"><img src="examples/output/gatys-katy_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/gatys-hoodwinked_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/gatys-lighthouse_X_kandinsky.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/spiderverse.png" width="100%"></td>
<td align="center"><img src="examples/output/gatys-katy_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/gatys-hoodwinked_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/gatys-lighthouse_X_spiderverse.webp" width="100%"></td>
</tr>
</table>

### Magenta - feed-forward arbitrary stylization

Ghiasi et al., Google Magenta (2017). A style-prediction sub-network compresses the style image into a low-dimensional embedding that parameterizes conditional instance-norm layers in a transfer network operating on the content image; the stylized image comes out in a single forward pass - no per-image optimization, no attention. That is why it returns in seconds, and also why it tends toward a smoother, more "averaged" stylization than the other two: a single global style vector cannot localize fine ornament or hard edges in the style image to specific regions of the content. Colour palettes transfer well; high-frequency style detail tends to dissolve into the content's textures rather than persist as discrete marks.

This model produces my favorite results, and this architectural direction is the route I would take to pursue arbitrary image style transfer further. Arbitrary-style GAN variants (AdaIN-WCT, MUNIT) would bloat this project size too much though, so I'll look into them separately.

The transfer network runs fully-convolutionally, so output resolution tracks the content image's aspect ratio up to a 2048 px longest-side cap.

<table>
<tr>
<td></td>
<td align="center" width="25%"><img src="examples/content/katy.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/hoodwinked.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/lighthouse.png" width="100%"></td>
</tr>
<tr>
<td align="center" width="25%"><img src="examples/style/ty.png" width="100%"></td>
<td align="center"><img src="examples/output/magenta-katy_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/magenta-hoodwinked_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/magenta-lighthouse_X_ty.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/kandinsky.png" width="100%"></td>
<td align="center"><img src="examples/output/magenta-katy_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/magenta-hoodwinked_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/magenta-lighthouse_X_kandinsky.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/spiderverse.png" width="100%"></td>
<td align="center"><img src="examples/output/magenta-katy_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/magenta-hoodwinked_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/magenta-lighthouse_X_spiderverse.webp" width="100%"></td>
</tr>
</table>

### StyTr² - transformer-based arbitrary style transfer

Deng et al. (CVPR 2022). A pure-transformer alternative to both CNN feed-forward (Magenta) and per-image optimization (Gatys). Content and style images are tokenized via a stride-8 patch embedding, encoded by separate transformer stacks with content-aware positional encoding (CAPE), and fused by a cross-attention decoder before a convolutional upsampler returns to image space. Because attention operates patch-wise rather than through a single global style code, fine style detail and content tonality (notably true blacks) survive better than in Magenta, while inference stays feed-forward - runtime sits between the other two at ~30 s on CPU at 512², bounded by the O(N²) attention over 64×64 = 4096 tokens. The price: the patch-grid reshape requires `H == W`, so wide content is centre-cropped to 512×512 before inference and loses its outer regions. Pretrained weights are pulled from the `datnguyentien204/Sty_TR2_38` Hugging Face mirror on first use.

Notes:
StyTr² prints the device it loaded onto on first call (CPU only on Apple Silicon). 

<table>
<tr>
<td></td>
<td align="center" width="25%"><img src="examples/content/katy.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/hoodwinked.png" width="100%"></td>
<td align="center" width="25%"><img src="examples/content/lighthouse.png" width="100%"></td>
</tr>
<tr>
<td align="center" width="25%"><img src="examples/style/ty.png" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-katy_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-hoodwinked_X_ty.webp" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-lighthouse_X_ty.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/kandinsky.png" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-katy_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-hoodwinked_X_kandinsky.webp" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-lighthouse_X_kandinsky.webp" width="100%"></td>
</tr>
<tr>
<td align="center"><img src="examples/style/spiderverse.png" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-katy_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-hoodwinked_X_spiderverse.webp" width="100%"></td>
<td align="center"><img src="examples/output/stytr2-lighthouse_X_spiderverse.webp" width="100%"></td>
</tr>
</table>
