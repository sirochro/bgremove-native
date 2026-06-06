"""Shared alpha matte post-processing used by both image (rembg) and video (RVM) tabs.

Order: low cut → high boost → expand/contract → Gaussian feather.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion


def postprocess_alpha(
    pha: np.ndarray,
    low: int,
    high: int,
    edge: int,
    feather: int,
) -> np.ndarray:
    """Apply alpha matte adjustments.

    Args:
        pha: (H, W) uint8 alpha matte.
        low: 0..255. pha < low → 0.
        high: 0..255. pha > high → 255. (255 = no-op)
        edge: -5..+5 pixels. + dilates, - erodes.
        feather: 0..20 pixels. Gaussian sigma = feather / 2.
    """
    out = pha
    if low > 0:
        out = np.where(out < low, np.uint8(0), out)
    if high < 255:
        out = np.where(out > high, np.uint8(255), out)
    if edge > 0:
        size = 2 * edge + 1
        out = grey_dilation(out, size=(size, size))
    elif edge < 0:
        size = -2 * edge + 1
        out = grey_erosion(out, size=(size, size))
    if feather > 0:
        out = gaussian_filter(out, sigma=feather / 2.0)
    return out.astype(np.uint8)


def is_default(low: int, high: int, edge: int, feather: int) -> bool:
    """True if all parameters are at no-op defaults; skip processing entirely."""
    return low == 0 and high == 255 and edge == 0 and feather == 0


def clamp_params(low: int, high: int, edge: int, feather: int) -> tuple[int, int, int, int]:
    """Clamp each parameter to its valid range."""
    return (
        max(0, min(255, int(low))),
        max(0, min(255, int(high))),
        max(-5, min(5, int(edge))),
        max(0, min(20, int(feather))),
    )
