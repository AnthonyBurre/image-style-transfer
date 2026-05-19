"""Image Analogies (Hertzmann, Jacobs, Oliver, Curless, Salesin — SIGGRAPH 2001).

A pre-neural, patch-based style transfer baseline, included as the historical
bookend before the dedicated neural-style-transfer era. Operates entirely in
pixel space: for each output pixel, search the style image for a 5×5 luminance
patch whose neighbourhood best matches the content image around that point,
then copy the corresponding style pixel into the output.

Two pieces beyond raw nearest-neighbour patch search make the result hold
together:

  * **YIQ luminance matching.** Patch similarity is computed on the Y channel
    only, so structure (not colour) drives the match. The style image's IQ
    channels are transferred through unchanged, which is what gives the output
    the style's palette.
  * **Ashikhmin coherence.** At each pixel, candidate sources include not just
    the global ANN match but also "shifted" coordinates from already-synthesised
    upper-left neighbours. The coherent candidate is preferred when its
    distance is within a κ-weighted multiple of the ANN distance (paper eq. 4).
    Without this term the output is patchy noise; with it, marks and brush
    strokes survive across neighbouring pixels.

A 3-level Gaussian pyramid threads coarse structure into the fine synthesis:
each level's source-coordinate map seeds the next level as an extra coherence
candidate. cKDTree builds and batched ANN queries handle the otherwise-quadratic
patch search.

No GPU path — pre-neural is pre-GPU as a matter of historical accuracy as much
as implementation cost.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from scipy.spatial import cKDTree

from ..image import prepare

LABEL = "Analogies (slow — patch-based, pre-neural)"
BLURB = "**Image Analogies** (Hertzmann et al., 2001) runs a multi-scale patch search — the pre-neural baseline, slow and softer than the neural methods"

PATCH_SIZE = 5
PYRAMID_LEVELS = 4
KAPPA = 10.0

# Runtime is roughly linear in output pixel count (one KDTree query plus a few
# patch comparisons per output pixel) and grows mildly with style pool size.
# 512² content × 512² style lands around a minute on CPU — the same order as
# Gatys and StyTr², which feels right for the "slow but historically faithful"
# framing.
CONTENT_MAX_DIM = 512
STYLE_MAX_DIM = 512

# ITU-R BT.601 RGB→YIQ. Y carries luminance (used for matching); IQ carry the
# chroma the style transfers into the output.
_RGB_TO_YIQ = np.array(
    [
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312],
    ],
    dtype=np.float32,
)
_YIQ_TO_RGB = np.linalg.inv(_RGB_TO_YIQ).astype(np.float32)

# 5-tap binomial kernel approximates a Gaussian with σ≈1 and is the standard
# choice for image pyramids (Burt & Adelson).
_BINOMIAL = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


def _to_yiq(rgb_uint8):
    return (rgb_uint8.astype(np.float32) / 255.0) @ _RGB_TO_YIQ.T


def _from_yiq(yiq):
    rgb = yiq @ _YIQ_TO_RGB.T
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _downsample(channel):
    blurred = convolve1d(channel, _BINOMIAL, axis=0, mode="reflect")
    blurred = convolve1d(blurred, _BINOMIAL, axis=1, mode="reflect")
    return blurred[::2, ::2]


def _build_pyramid(channel, levels):
    """Returns ``[finest, ..., coarsest]`` (index 0 is the original resolution)."""
    pyr = [channel]
    for _ in range(levels - 1):
        pyr.append(_downsample(pyr[-1]))
    return pyr


def _extract_patches(channel, patch_size):
    """``(H, W) → (H, W, patch_size²)`` reflect-padded flattened patches."""
    half = patch_size // 2
    padded = np.pad(channel, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (patch_size, patch_size))
    h, w = channel.shape
    return windows.reshape(h, w, patch_size * patch_size)


def _synthesize_level(
    content_y,
    style_y,
    style_iq,
    parent_source_y,
    parent_source_x,
    coherence_weight,
    progress,
    progress_start,
    progress_end,
    level_label,
):
    h, w = content_y.shape
    hs, ws = style_y.shape

    content_patches = _extract_patches(content_y, PATCH_SIZE)
    style_patches_grid = _extract_patches(style_y, PATCH_SIZE)
    style_patches_flat = style_patches_grid.reshape(hs * ws, -1)

    tree = cKDTree(style_patches_flat)

    queries = content_patches.reshape(h * w, -1)
    ann_dists, ann_flat_idxs = tree.query(queries)
    ann_y = (ann_flat_idxs // ws).reshape(h, w).astype(np.int32)
    ann_x = (ann_flat_idxs % ws).reshape(h, w).astype(np.int32)
    ann_d2 = (ann_dists.astype(np.float32) ** 2).reshape(h, w)

    source_y = np.full((h, w), -1, dtype=np.int32)
    source_x = np.full((h, w), -1, dtype=np.int32)
    output_y = np.empty((h, w), dtype=np.float32)
    output_iq = np.empty((h, w, 2), dtype=np.float32)

    # Causal (upper-left) coherence neighbours, in raster order.
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1))

    has_parent = parent_source_y is not None

    for r in range(h):
        for c in range(w):
            content_q = content_patches[r, c]

            best_coh_d2 = np.inf
            best_coh_y = -1
            best_coh_x = -1

            for dr, dc in neighbours:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nc < 0 or nc >= w:
                    continue
                if source_y[nr, nc] < 0:
                    continue
                cand_y = source_y[nr, nc] - dr
                cand_x = source_x[nr, nc] - dc
                if cand_y < 0 or cand_y >= hs or cand_x < 0 or cand_x >= ws:
                    continue
                diff = style_patches_grid[cand_y, cand_x] - content_q
                d2 = float(np.dot(diff, diff))
                if d2 < best_coh_d2:
                    best_coh_d2 = d2
                    best_coh_y = cand_y
                    best_coh_x = cand_x

            # Parent (coarser-level) source map contributes one more candidate.
            if has_parent:
                pr = r >> 1
                pc = c >> 1
                cand_y = parent_source_y[pr, pc] * 2 + (r & 1)
                cand_x = parent_source_x[pr, pc] * 2 + (c & 1)
                if 0 <= cand_y < hs and 0 <= cand_x < ws:
                    diff = style_patches_grid[cand_y, cand_x] - content_q
                    d2 = float(np.dot(diff, diff))
                    if d2 < best_coh_d2:
                        best_coh_d2 = d2
                        best_coh_y = cand_y
                        best_coh_x = cand_x

            ann_d2_here = ann_d2[r, c]
            if best_coh_y >= 0 and best_coh_d2 <= ann_d2_here * coherence_weight:
                sy = best_coh_y
                sx = best_coh_x
            else:
                sy = int(ann_y[r, c])
                sx = int(ann_x[r, c])

            source_y[r, c] = sy
            source_x[r, c] = sx
            output_y[r, c] = style_y[sy, sx]
            output_iq[r, c] = style_iq[sy, sx]

        if progress is not None:
            frac = progress_start + (progress_end - progress_start) * (r + 1) / h
            progress(frac, desc=f"{level_label} row {r + 1}/{h}")

    return output_y, output_iq, source_y, source_x


def stylize(content_image, style_image, *, progress=None):
    if content_image is None or style_image is None:
        return None

    content_rgb = np.asarray(prepare(content_image, CONTENT_MAX_DIM))
    style_rgb = np.asarray(prepare(style_image, STYLE_MAX_DIM))

    content_yiq = _to_yiq(content_rgb)
    style_yiq = _to_yiq(style_rgb)

    content_pyr = _build_pyramid(content_yiq[..., 0], PYRAMID_LEVELS)
    style_y_pyr = _build_pyramid(style_yiq[..., 0], PYRAMID_LEVELS)
    style_i_pyr = _build_pyramid(style_yiq[..., 1], PYRAMID_LEVELS)
    style_q_pyr = _build_pyramid(style_yiq[..., 2], PYRAMID_LEVELS)

    # Progress slices weighted by output pixel count so the finest level (which
    # dominates wall time) gets the longest slice.
    pixel_counts = [p.size for p in content_pyr]
    total_px = float(sum(pixel_counts))

    parent_sy = None
    parent_sx = None
    output_y = None
    output_iq = None

    # Iterate coarse → fine. content_pyr[0] is the finest, so reverse.
    progress_acc = 0.0
    for l in range(PYRAMID_LEVELS - 1, -1, -1):
        c_y = content_pyr[l]
        s_y = style_y_pyr[l]
        s_iq = np.stack([style_i_pyr[l], style_q_pyr[l]], axis=-1)

        # Paper eq. (4): coherence weight is 1 + κ·2^(L - L_max).
        # L_max = 0 (finest); l = level index in the pyramid (0 is finest, larger is coarser).
        coherence_weight = 1.0 + KAPPA * (2.0 ** (-l))

        level_frac = pixel_counts[l] / total_px
        p_start = progress_acc
        p_end = progress_acc + level_frac
        progress_acc = p_end
        level_label = f"L{PYRAMID_LEVELS - l}/{PYRAMID_LEVELS}"

        output_y, output_iq, parent_sy, parent_sx = _synthesize_level(
            c_y,
            s_y,
            s_iq,
            parent_sy,
            parent_sx,
            coherence_weight,
            progress,
            p_start,
            p_end,
            level_label,
        )

    final_yiq = np.concatenate([output_y[..., None], output_iq], axis=-1)
    return Image.fromarray(_from_yiq(final_yiq))
