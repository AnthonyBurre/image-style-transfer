"""Optimization-based neural style transfer (Gatys, Ecker & Bethge, 2015) via
VGG19 features.

Unlike Magenta, this does not "infer" a stylized image in one shot. Instead it
iteratively edits an output image so that:

  * its high-level VGG19 features (block5_conv2) match the *content* image, and
  * the Gram matrices of its low/mid-level VGG19 features match those of the
    *style* image.

That formulation matches *texture statistics* rather than copying style-image
patches, which is why it produces much more abstracted/painterly results than
Magenta — at the cost of running a full optimisation loop per request.
"""
import numpy as np
import tensorflow as tf
from PIL import Image

from ..image import prepare

LABEL = "Gatys VGG19 (slow — minutes, more abstracted)"
BLURB = "**Gatys VGG19** runs an optimisation loop (slow, but produces more abstracted/painterly stylisation)"

CONTENT_LAYER = "block5_conv2"
STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
TV_WEIGHT = 30.0

# VGG19 fwd+bwd on CPU is heavy; 512 keeps a single optimisation step under a
# few seconds and fits comfortably in the 4 GB container.
MAX_DIM = 512
DEFAULT_STEPS = 300
LEARNING_RATE = 0.02

_extractor = None


def _build_extractor():
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS + [CONTENT_LAYER]]
    return tf.keras.Model(vgg.input, outputs)


def _get_extractor():
    global _extractor
    if _extractor is None:
        print("Loading VGG19 (first run downloads ~80 MB of weights)...")
        _extractor = _build_extractor()
        print("VGG19 loaded.")
    return _extractor


def _to_tensor(image, max_dim):
    arr = np.asarray(prepare(image, max_dim), dtype=np.float32) / 255.0
    return tf.convert_to_tensor(arr)[tf.newaxis, :]


def _to_pil(tensor):
    arr = (np.array(tensor) * 255.0).astype(np.uint8)
    if arr.ndim > 3:
        arr = arr[0]
    return Image.fromarray(arr)


def _gram_matrix(feature):
    # Normalising by H*W keeps the Gram magnitudes layer-comparable.
    result = tf.linalg.einsum("bijc,bijd->bcd", feature, feature)
    n = tf.cast(tf.shape(feature)[1] * tf.shape(feature)[2], tf.float32)
    return result / n


def _features(image, extractor):
    # VGG19 was trained on ImageNet-mean-subtracted BGR pixels in [0, 255];
    # ``preprocess_input`` does that conversion. Our ``image`` lives in [0, 1].
    preprocessed = tf.keras.applications.vgg19.preprocess_input(image * 255.0)
    outputs = extractor(preprocessed)
    style_features = outputs[: len(STYLE_LAYERS)]
    content_feature = outputs[len(STYLE_LAYERS)]
    style_grams = [_gram_matrix(s) for s in style_features]
    return style_grams, content_feature


def stylize(content_image, style_image, *, steps=DEFAULT_STEPS, progress=None):
    if content_image is None or style_image is None:
        return None

    extractor = _get_extractor()

    content_tensor = _to_tensor(content_image, MAX_DIM)
    style_tensor = _to_tensor(style_image, MAX_DIM)

    target_style_grams, _ = _features(style_tensor, extractor)
    _, target_content = _features(content_tensor, extractor)

    image = tf.Variable(content_tensor)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=0.99, epsilon=1e-1
    )

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            style_grams, content_feature = _features(image, extractor)
            style_loss = tf.add_n(
                [tf.reduce_mean((g - t) ** 2) for g, t in zip(style_grams, target_style_grams)]
            ) * (STYLE_WEIGHT / len(STYLE_LAYERS))
            content_loss = (
                tf.reduce_mean((content_feature - target_content) ** 2) * CONTENT_WEIGHT
            )
            tv_loss = tf.reduce_sum(tf.image.total_variation(image)) * TV_WEIGHT
            loss = style_loss + content_loss + tv_loss
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        return loss

    print(f"Running Gatys optimization for {steps} steps (slow on CPU)...")
    for step in range(steps):
        loss = train_step()
        if progress is not None:
            progress((step + 1) / steps, desc=f"step {step + 1}/{steps}")
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step + 1}/{steps}, loss = {float(loss):.0f}")

    return _to_pil(image)
