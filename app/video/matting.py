"""RVM (Robust Video Matting) ONNX inference with DirectML.

Recurrent model: r1..r4 hidden states carry between frames in the same clip
for temporal consistency. Reset between independent clips or at scene cuts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class RVMSession:
    """Per-clip RVM session. Holds recurrent state."""

    def __init__(self, model_path: str | Path, device_id: int = 0):
        providers = [
            ("DmlExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.reset()

    def reset(self) -> None:
        z = np.zeros((1, 1, 1, 1), dtype=np.float32)
        self.r1, self.r2, self.r3, self.r4 = z, z, z, z

    def infer(
        self, rgb: np.ndarray, downsample_ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one frame.

        Args:
            rgb: (H, W, 3) uint8 in RGB order.
            downsample_ratio: see ``suggest_downsample_ratio``.

        Returns:
            fgr: (H, W, 3) uint8 — foreground RGB.
            pha: (H, W) uint8 — alpha matte.
        """
        src = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        dsr = np.asarray([downsample_ratio], dtype=np.float32)
        fgr, pha, self.r1, self.r2, self.r3, self.r4 = self.session.run(
            None,
            {
                "src": src,
                "r1i": self.r1,
                "r2i": self.r2,
                "r3i": self.r3,
                "r4i": self.r4,
                "downsample_ratio": dsr,
            },
        )
        fgr_u8 = (fgr[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        pha_u8 = (pha[0, 0] * 255.0).clip(0, 255).astype(np.uint8)
        return fgr_u8, pha_u8


def suggest_downsample_ratio(width: int, height: int) -> float:
    """Recommended downsample_ratio per RVM authors.

    SD: 1.0, 720p: 0.375, 1080p: 0.25, 4K: 0.125.
    """
    long_edge = max(width, height)
    if long_edge >= 3000:
        return 0.125
    if long_edge >= 1500:
        return 0.25
    if long_edge >= 1000:
        return 0.375
    return 1.0
