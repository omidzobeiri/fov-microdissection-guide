"""Write a small synthetic session folder to try fov_guide.py without real data.

    python examples/make_synthetic_session.py demo_session
    python fov_guide.py demo_session --dot-spacing-mm 0.93
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile


def make_session(folder: Path, dot_px: int = 93, grid_deg: float = 3.0) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    h, w = 1800, 2400
    img = np.full((h, w, 3), 225, np.uint8)

    # Dotted sizing guide, turned by grid_deg.
    a, c = np.radians(grid_deg), np.array([w / 2, h / 2])
    for i in range(-20, 21):
        for j in range(-20, 21):
            p = c + dot_px * np.array([i * np.cos(a) - j * np.sin(a), i * np.sin(a) + j * np.cos(a)])
            if 0 < p[0] < w and 0 < p[1] < h:
                cv2.circle(img, (int(p[0]), int(p[1])), 15, (40, 40, 40), -1)

    # Semi-transparent beige slab.
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (1200, 900), (620, 480), 10, 0, 360, 255, -1)
    tissue = img.copy()
    tissue[mask > 0] = (205, 180, 130)
    img = cv2.addWeighted(img, 0.15, tissue, 0.85, 0)
    cv2.imwrite(str(folder / "slab.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Warped FOV channels: channel 0 = trapezoid + FOV square, channel 1 empty.
    ch = np.zeros((2, h, w), np.uint8)
    trap = np.array([[1000, 700], [1450, 720], [1400, 1150], [1050, 1100]], np.int32)
    cv2.polylines(ch[0], [trap], True, 255, 4)
    cv2.rectangle(ch[0], (1120, 820), (1320, 1000), 255, 4)
    tifffile.imwrite(folder / "slab_trap_warped.tif", ch)


if __name__ == "__main__":
    make_session(Path(sys.argv[1] if len(sys.argv) > 1 else "demo_session"))
