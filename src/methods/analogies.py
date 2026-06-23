"""Image Analogies (Hertzmann, Jacobs, Oliver, Curless, Salesin — SIGGRAPH 2001) §4.5.

The paper's *artistic filters* application of the image-analogies framework:
given a textured style image A′ and a content image B, synthesise an output
B′ that stands to B as A′ stands to A — i.e. it learns the analogy
``A : A′ :: B : B′``.

The wrinkle for our setup is that we never receive A — the user only ever
uploads a single style image (the textured A′). The paper anticipates this
("for many example images, we do not have a source photograph available")
and prescribes synthesising A from A′ by edge-preserving smoothing — they
used Photoshop's *Smart Blur*; we use **Perona–Malik anisotropic diffusion**,
which kills fine texture while keeping sharp contours and is implementable
with numpy alone. The result is the "untextured" example the algorithm needs
to anchor structural matches against.

The synthesis is a multi-scale patch search in pixel space. At each output
pixel we build a feature vector with two halves:

  * **Source half (matched against A).** A full 5×5 luminance patch in the
    content image B is compared against full 5×5 patches in the smoothed A.
    This carries structural similarity — "what does the *shape* here look
    like?" — without being distracted by A′'s texture.
  * **Target half (matched against A′).** A causal 5×5 patch from the
    already-synthesised B′ is compared against the same causal positions in
    A′. This is the term that keeps the textured output self-consistent —
    brush strokes survive across neighbouring pixels because each pixel's
    match has to agree with what its already-painted neighbours pulled from
    the style.

Once a source coordinate is chosen the textured A′ pixel is copied into B′
(matching uses A, painting uses A′ — the §4.5 split).

Two further pieces beyond raw nearest-neighbour patch search make the result
hold together:

  * **YIQ luminance matching.** All patch comparisons happen on the Y channel
    only, so structure (not colour) drives the match. A′'s IQ channels are
    transferred through unchanged, which is what gives B′ the style's palette.
  * **Ashikhmin coherence.** At each pixel, candidate sources include not just
    the global ANN match but also "shifted" coordinates inherited from
    already-synthesised upper-left neighbours and from the parent (coarser)
    pyramid level. The coherent candidate is preferred when its distance is
    within a κ-weighted multiple of the ANN distance (paper eq. 4).

A 4-level Gaussian pyramid threads coarse structure into the fine synthesis:
each level's source-coordinate map seeds the next level as an extra coherence
candidate. cKDTree builds handle the otherwise-quadratic source-side patch
search; queries run per-pixel because the target half of the feature depends
on what's already been synthesised.

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

# Perona–Malik anisotropic diffusion parameters for synthesising A from A′.
# K is the gradient threshold separating "edge" from "texture" in [0,1]-
# normalised luminance — gradients much larger than K survive, smaller ones
# get smoothed. λ is the per-iteration step size; with 4-neighbour stencil the
# CFL bound is 0.25, so 0.2 is safely stable.
DIFFUSION_ITERS = 15
DIFFUSION_K = 0.1
DIFFUSION_LAMBDA = 0.2

# Weight on the target-side (A′/B′) half of the matching feature vector.
# At 1.0 the structural and textural halves contribute equally; lowering it
# biases the match toward content fidelity at the cost of stroke coherence.
TARGET_HALF_WEIGHT = 1.0

# Per-position Gaussian weighting on patches before computing distances —
# centre pixel and its immediate neighbours dominate the match; corner
# luminance coincidences contribute less. σ chosen so the corner of a 5×5
# patch has ~0.1 the weight of the centre.
PATCH_SIGMA = 1.2

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


def _anisotropic_diffuse(channel):
    """Perona–Malik anisotropic diffusion — stand-in for the paper's Smart Blur.

    Edge-preserving smoothing on a single luminance channel. Each iteration
    nudges every pixel toward its neighbours, weighted by an exponential
    conductance ``g(|∇|) = exp(-(∇/K)²)`` so that sharp contours (large
    gradient) are barely diffused while flat-but-textured regions converge
    toward their local mean.
    """
    u = channel.astype(np.float32, copy=True)
    for _ in range(DIFFUSION_ITERS):
        # Forward differences in the four cardinal directions; out-of-bounds
        # neighbours contribute zero (Neumann boundary).
        north = np.zeros_like(u)
        south = np.zeros_like(u)
        east = np.zeros_like(u)
        west = np.zeros_like(u)
        north[1:, :] = u[:-1, :] - u[1:, :]
        south[:-1, :] = u[1:, :] - u[:-1, :]
        east[:, :-1] = u[:, 1:] - u[:, :-1]
        west[:, 1:] = u[:, :-1] - u[:, 1:]

        inv_k2 = 1.0 / (DIFFUSION_K * DIFFUSION_K)
        gn = np.exp(-(north * north) * inv_k2)
        gs = np.exp(-(south * south) * inv_k2)
        ge = np.exp(-(east * east) * inv_k2)
        gw = np.exp(-(west * west) * inv_k2)

        u += DIFFUSION_LAMBDA * (gn * north + gs * south + ge * east + gw * west)
    return u


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


def _gaussian_patch_weights(patch_size, sigma):
    """Flat 2D Gaussian weights for patch-distance scaling.

    Multiplied into patch vectors before forming the matching feature, so
    L2 distance becomes ``Σ (g·a − g·b)² = Σ g²·(a − b)²`` — i.e. centre
    pixels contribute quadratically more than corners. Normalised so the
    centre weight is 1; absolute scale just rescales every distance and
    cancels in any inequality comparison.
    """
    coords = np.arange(patch_size) - patch_size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    w = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return (w / w.max()).reshape(-1).astype(np.float32)


def _causal_mask(patch_size):
    """Flat boolean mask selecting the strictly upper-left half of a patch.

    For raster-order synthesis, a "causal" patch covers only positions that
    have already been written: rows above the centre row in full, plus the
    pixels to the left of centre on the centre row. Used for both the
    A′-side feature (so source and target halves carry the same dimensions)
    and the B′-side feature (so we read only already-synthesised pixels).
    """
    half = patch_size // 2
    mask = np.zeros((patch_size, patch_size), dtype=bool)
    mask[:half, :] = True
    mask[half, :half] = True
    return mask.reshape(-1)


def _synthesize_level(
    content_y,
    style_a_y,
    style_y,
    style_iq,
    parent_source_y,
    parent_source_x,
    parent_output_y,
    coherence_weight,
    progress,
    progress_start,
    progress_end,
    level_label,
):
    h, w = content_y.shape
    hs, ws = style_a_y.shape
    half = PATCH_SIZE // 2
    feat_dim = PATCH_SIZE * PATCH_SIZE

    # Static source-side feature for every A′ pixel: full A patch (structural,
    # untextured) concatenated with the causal half of the A′ patch (textural).
    # Non-causal positions in the A′ half are zeroed; we apply the same mask
    # to the B′ side of the query at runtime, so the L2 distance only sums
    # contributions over causal positions on either side. A Gaussian centred
    # on the patch midpoint scales all positions so centre pixels dominate.
    g_weights = _gaussian_patch_weights(PATCH_SIZE, PATCH_SIGMA)
    content_patches = _extract_patches(content_y, PATCH_SIZE) * g_weights
    a_patches = _extract_patches(style_a_y, PATCH_SIZE) * g_weights
    a_prime_patches = _extract_patches(style_y, PATCH_SIZE) * g_weights

    causal = _causal_mask(PATCH_SIZE).astype(np.float32) * TARGET_HALF_WEIGHT
    a_prime_causal = a_prime_patches * causal

    source_feat = np.concatenate(
        [a_patches.reshape(hs * ws, feat_dim), a_prime_causal.reshape(hs * ws, feat_dim)],
        axis=1,
    )
    tree = cKDTree(source_feat)

    source_y = np.full((h, w), -1, dtype=np.int32)
    source_x = np.full((h, w), -1, dtype=np.int32)
    output_y = np.empty((h, w), dtype=np.float32)
    output_iq = np.empty((h, w, 2), dtype=np.float32)

    # Padded scratch buffer for reading already-written B′ neighbourhoods
    # without bounds checks. At all but the coarsest level we initialise from
    # the upsampled output of the previous (coarser) level — paper-canonical
    # coarse-to-fine *refinement*, not just seeding. That way the causal-B′
    # patch around an early raster pixel contains real luminance from the
    # coarser pass rather than zero, and the ‖B′_causal − A′_causal‖² term
    # contributes genuine signal instead of dark-biased noise.
    if parent_output_y is not None:
        padded_out = np.pad(parent_output_y, half, mode="reflect").astype(np.float32, copy=False)
    else:
        padded_out = np.zeros((h + 2 * half, w + 2 * half), dtype=np.float32)

    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1))
    has_parent = parent_source_y is not None

    for r in range(h):
        for c in range(w):
            # Dynamic target-side query: full B patch + causal half of the
            # in-progress B′ patch around (r, c). content_patches is already
            # Gaussian-weighted; multiply the live B′ readout by the same
            # weights (and the causal mask, which carries TARGET_HALF_WEIGHT).
            out_window = padded_out[r:r + PATCH_SIZE, c:c + PATCH_SIZE].reshape(-1)
            query = np.empty(2 * feat_dim, dtype=np.float32)
            query[:feat_dim] = content_patches[r, c]
            query[feat_dim:] = out_window * g_weights * causal

            ann_dist, ann_idx = tree.query(query, k=1)
            ann_d2 = float(ann_dist) ** 2
            ann_sy = int(ann_idx) // ws
            ann_sx = int(ann_idx) % ws

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
                diff = source_feat[cand_y * ws + cand_x] - query
                d2 = float(np.dot(diff, diff))
                if d2 < best_coh_d2:
                    best_coh_d2 = d2
                    best_coh_y = cand_y
                    best_coh_x = cand_x

            # Parent (coarser-level) source map contributes one more candidate.
            if has_parent:
                pr = r >> 1
                pc = c >> 1
                cand_y = int(parent_source_y[pr, pc]) * 2 + (r & 1)
                cand_x = int(parent_source_x[pr, pc]) * 2 + (c & 1)
                if 0 <= cand_y < hs and 0 <= cand_x < ws:
                    diff = source_feat[cand_y * ws + cand_x] - query
                    d2 = float(np.dot(diff, diff))
                    if d2 < best_coh_d2:
                        best_coh_d2 = d2
                        best_coh_y = cand_y
                        best_coh_x = cand_x

            if best_coh_y >= 0 and best_coh_d2 <= ann_d2 * coherence_weight:
                sy = best_coh_y
                sx = best_coh_x
            else:
                sy = ann_sy
                sx = ann_sx

            source_y[r, c] = sy
            source_x[r, c] = sx
            output_y[r, c] = style_y[sy, sx]
            output_iq[r, c] = style_iq[sy, sx]
            padded_out[r + half, c + half] = output_y[r, c]

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

    # A is synthesised from A′ by anisotropic-diffusion smoothing of luminance.
    # Stands in for the paper's "Smart Blur" preprocessing — gives us the
    # untextured half of the A : A′ pair without asking the user for one.
    style_a_y = _anisotropic_diffuse(style_yiq[..., 0])

    content_pyr = _build_pyramid(content_yiq[..., 0], PYRAMID_LEVELS)
    style_y_pyr = _build_pyramid(style_yiq[..., 0], PYRAMID_LEVELS)
    style_a_pyr = _build_pyramid(style_a_y, PYRAMID_LEVELS)
    style_i_pyr = _build_pyramid(style_yiq[..., 1], PYRAMID_LEVELS)
    style_q_pyr = _build_pyramid(style_yiq[..., 2], PYRAMID_LEVELS)

    # Progress slices weighted by output pixel count so the finest level (which
    # dominates wall time) gets the longest slice.
    pixel_counts = [p.size for p in content_pyr]
    total_px = float(sum(pixel_counts))

    parent_sy = None
    parent_sx = None
    parent_output_y = None  # upsampled luminance of coarser-level B′ output
    output_y = None
    output_iq = None

    # Iterate coarse → fine. content_pyr[0] is the finest, so reverse.
    progress_acc = 0.0
    for l in range(PYRAMID_LEVELS - 1, -1, -1):
        c_y = content_pyr[l]
        s_a_y = style_a_pyr[l]
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
            s_a_y,
            s_y,
            s_iq,
            parent_sy,
            parent_sx,
            parent_output_y,
            coherence_weight,
            progress,
            p_start,
            p_end,
            level_label,
        )

        # Upsample this level's luminance output by 2× for the next finer
        # level's B′ initialisation. Nearest-neighbour repeat then crop to
        # the exact next-level shape — the next finer level only needs the
        # readout to be roughly in the right ballpark, not pixel-perfect.
        if l > 0:
            next_h, next_w = content_pyr[l - 1].shape
            up = np.repeat(np.repeat(output_y, 2, axis=0), 2, axis=1)
            parent_output_y = up[:next_h, :next_w]

    final_yiq = np.concatenate([output_y[..., None], output_iq], axis=-1)
    return Image.fromarray(_from_yiq(final_yiq))
