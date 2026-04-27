"""Minimal helper vendored from StyTR-2's ``ViT_helper.py``.

The original file imported ``torch._six.container_abcs``, which was removed
in PyTorch 1.8+. ``PatchEmbed`` only ever uses ``to_2tuple`` from this module,
so the rest (DropPath, trunc_normal_, etc.) is dropped.
"""
from collections.abc import Iterable
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)
