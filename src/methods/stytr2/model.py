"""StyTR-2 model definition (inference-only).

Adapted from https://github.com/diyiiyiii/StyTR-2/blob/main/models/StyTR.py.
Two simplifications relative to the original:

* The original ``StyTrans`` module also wires up a frozen VGG-19 encoder used
  *only* for computing the training losses (content + style + identity). For
  inference it is dead weight, so the VGG construction and all loss methods
  are dropped.
* The original ``forward`` accepts ``NestedTensor`` to support ragged batches.
  We always feed a single content + single style tensor, so the NestedTensor
  plumbing (and its ``util.box_ops`` / ``util.misc`` dependencies) is gone.

The ``decoder`` and ``PatchEmbed`` definitions are byte-for-byte the same as
upstream so the released ``decoder_iter_160000.pth`` and
``embedding_iter_160000.pth`` checkpoints load cleanly.
"""
import torch.nn as nn

from .vit_helper import to_2tuple


class PatchEmbed(nn.Module):
    """Image-to-patch embedding via a strided conv (8x8 patches by default)."""

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.proj(x)


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class StyTransInference(nn.Module):
    """Inference-only StyTR-2: PatchEmbed → cross-attention transformer → CNN decoder."""

    def __init__(self, embedding, transformer, decoder_module):
        super().__init__()
        self.embedding = embedding
        self.transformer = transformer
        self.decode = decoder_module

    def forward(self, content, style):
        content_tokens = self.embedding(content)
        style_tokens = self.embedding(style)
        hs = self.transformer(style_tokens, None, content_tokens, None, None)
        return self.decode(hs)
