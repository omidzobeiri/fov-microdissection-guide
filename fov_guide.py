#!/usr/bin/env python3
"""
fov_guide.py - Make a 1:1 microdissection guide (SVG) for a tangential brain slab.

The guide shows, at true physical size (mm):
  * the slab boundary (automatic first pass, refine it in Inkscape),
  * the two-photon (2p) field-of-view (FOV) square for each imaging channel,
  * the trapezoid cut outline around each FOV, with one edge turned to make the
    shape asymmetric (this keeps the orientation of the block), and
  * the cut lines: each trapezoid edge extended as a straight line across the
    whole guide (black, 0.176 mm), as in the guide template.

Inputs (one session folder)
  <name>.jpg              Stereo-microscope image of the slab on the dotted
                          sizing guide (protocol step 7).
  <name>_trap_warped.tif  BigWarp output: the FOV-outline channels of the
                          full-field 2p image, warped into the coordinates of
                          the slab image (protocol step 8.1). One channel per
                          FOV. Each channel holds the FOV square and the
                          trapezoid outline around it.

Outputs (written into the same folder)
  <folder>_fov_guide.svg            The guide, in mm (1 SVG unit = 1 mm).
  <folder>_fov_guide_aligned.jpg    Slab image, rotated to the grid and scaled
                                    to the SVG. Its DPI is set so that it
                                    imports into Inkscape at the same size.
  <folder>_fov_guide_aligned_overlay.jpg
                                    Aligned image with the warped outlines.
  brain_boundary_overlay.png        QC: automatic slab boundary on the image.
  svg_preview.png                   QC: preview of the SVG content.

Usage
  python fov_guide.py /path/to/session_folder --dot-spacing-mm 0.93
  python fov_guide.py                  # opens a folder-picker dialog
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree

__version__ = "1.0.0"

# Resolution of the internal working space. 100 px = 1 mm.
PX_PER_MM = 100.0
ALIGNED_DPI = PX_PER_MM * 25.4  # 2540 DPI, so 1 px = 1/PX_PER_MM mm

# One color per FOV channel: (SVG color name, RGB).
FOV_COLORS = [
    ("green", (0, 255, 0)),
    ("blue", (0, 0, 255)),
    ("orange", (255, 165, 0)),
    ("purple", (128, 0, 128)),
    ("cyan", (0, 255, 255)),
]


@dataclass
class GuideParams:
    """User parameters. The defaults are the values used in the protocol."""

    dot_spacing_mm: float = 0.93
    """Real distance between adjacent dots of the printed sizing guide (mm).
    Measure your printed guide; printers often do not print at exactly 100%."""

    area_scale: float = 1.0
    """Scale of the trapezoid area about its center (1.0 = no change)."""

    width_scale: float = 1.0
    """Scale across the two parallel edges of the trapezoid only
    (>1 wider, <1 narrower). Applied after area_scale."""

    angle_increase_deg: float = 8.0
    """Extra rotation (degrees) of the trapezoid edge that is most oblique to
    the FOV square. This makes the asymmetry easier to see under the scope."""

    edge_extend_factor: float = 0.0
    """Length of each cut line on each side of a trapezoid edge, as a multiple
    of the edge length. 0 = extend each line to the edges of the guide (the
    template layout: four straight lines across the whole guide)."""

    cut_color: str = "black"
    cut_width_mm: float = 0.176
    """Color and width of the cut lines. 0.176 mm = 0.5 pt, as in the template."""

    brain_smooth_sigma: float = 0.012
    """Smoothing of the automatic slab boundary, as a fraction of the contour
    length. 0.012 follows real concavities; 0.10 gives a featureless oval."""

    fluo_low: int = 36000
    fluo_high: int = 65000
    """Intensity window used when the warped TIFF holds raw 16-bit FOV images
    instead of binary outlines."""

    channels: list[int] | None = None
    """TIFF channels (0-based) that hold FOV outlines. None = all channels."""

    plot: bool = True
    """Write the QC images."""


@dataclass
class FOV:
    ch_idx: int
    trapezoid: np.ndarray            # (N, 2) px, aligned space
    square: np.ndarray | None        # (M, 2) px, aligned space
    cut_lines: list = field(default_factory=list)  # [(p1, p2), ...] px
    color_name: str = "green"
    color_rgb: tuple = (0, 255, 0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def smooth_contour(contour: np.ndarray, sigma_frac: float, n_points: int = 1000) -> np.ndarray:
    """Resample a closed contour uniformly along arc length, then smooth the
    coordinates with a circular Gaussian (sigma = sigma_frac * n_points)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts_closed = np.vstack([pts, pts[:1]])
    seg = np.sqrt((np.diff(pts_closed, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1:
        return contour
    t = np.linspace(0, cum[-1], n_points, endpoint=False)
    x = np.interp(t, cum, pts_closed[:, 0])
    y = np.interp(t, cum, pts_closed[:, 1])
    sigma = sigma_frac * n_points
    x = gaussian_filter1d(x, sigma=sigma, mode="wrap")
    y = gaussian_filter1d(y, sigma=sigma, mode="wrap")
    return np.column_stack([x, y]).astype(np.int32).reshape(-1, 1, 2)


def closed_bezier_path(pts: np.ndarray, tension: float = 0.35) -> str:
    """Closed SVG path through pts (Catmull-Rom converted to cubic Bezier)."""
    n = len(pts)
    path = f"M {pts[0, 0]:.3f},{pts[0, 1]:.3f}"
    if n < 3:
        for p in pts[1:]:
            path += f" L {p[0]:.3f},{p[1]:.3f}"
        return path + " Z"
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        c1 = p1 + (p2 - p0) * tension / 3.0
        c2 = p2 - (p3 - p1) * tension / 3.0
        path += f" C {c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {p2[0]:.3f},{p2[1]:.3f}"
    return path + " Z"


def polygon_path(pts: np.ndarray) -> str:
    path = f"M {pts[0, 0]:.3f},{pts[0, 1]:.3f}"
    for p in pts[1:]:
        path += f" L {p[0]:.3f},{p[1]:.3f}"
    return path + " Z"


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(-1, 2).astype(np.float64)
    return (M @ np.hstack([pts, np.ones((len(pts), 1))]).T).T


def find_quadrilateral(contour: np.ndarray):
    """Approximate the convex hull of a contour by a 4-sided polygon.
    Returns (polygon or None, hull)."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx, hull
    if len(approx) == 5:
        # Remove the vertex after the shortest edge.
        pts = approx.reshape(-1, 2)
        lens = [np.linalg.norm(pts[(i + 1) % 5] - pts[i]) for i in range(5)]
        drop = (int(np.argmin(lens)) + 1) % 5
        keep = [i for i in range(5) if i != drop]
        return pts[keep].reshape(-1, 1, 2).astype(np.int32), hull
    if 4 <= len(approx) <= 6:
        return approx, hull
    return None, hull


# ---------------------------------------------------------------------------
# Step 1: slab image - boundary and dot grid
# ---------------------------------------------------------------------------

def detect_slab_mask(rgb: np.ndarray, sigma_frac: float) -> np.ndarray:
    """Binary mask of the slab. The tissue is beige on a light gray/white
    background, so it has higher color saturation."""
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    sat = cv2.GaussianBlur(sat, (71, 71), 0)
    otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(sat, int(otsu * 0.80), 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)

    # Fill holes: flood the background from the corner, then invert.
    flood = mask.copy()
    h, w = flood.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood)

    # Keep the largest connected component.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == largest) * 255).astype(np.uint8)

    # Light blur to remove the staircase edges from the morphology.
    mask = (cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0) > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = smooth_contour(max(contours, key=cv2.contourArea), sigma_frac)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [contour], -1, 255, -1)
    return out, contour


def detect_grid_dots(gray: np.ndarray) -> np.ndarray:
    """Centers (x, y) of the dots of the printed sizing guide."""
    h, w = gray.shape
    local_mean = cv2.GaussianBlur(gray.astype(np.float64), (151, 151), 0)
    dark = ((gray.astype(np.float64) / (local_mean + 1)) < 0.70).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k, iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k, iterations=1)

    n, _, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]

    # Typical dot area: median of blobs that are larger than noise.
    expected_spacing = min(h, w) / 12
    noise_area = max(50, (expected_spacing / 8) ** 2)
    real = areas[areas > noise_area]
    if len(real) > 5:
        dot_area = np.median(real)
    else:
        hist, bins = np.histogram(areas, bins=50)
        dot_area = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    a_min, a_max = max(dot_area * 0.25, noise_area), dot_area * 4.0

    centers = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if a_min < a < a_max and max(bw, bh) / (min(bw, bh) + 1e-6) < 2.5:
            centers.append(centroids[i])
    centers = np.array(centers)

    if len(centers) <= 5:
        return centers

    # Keep dots that have neighbors at the grid spacing.
    dists, _ = KDTree(centers).query(centers, k=min(7, len(centers)))
    nn = np.median(dists[:, 1])
    n_nb = ((dists[:, 1:] > 0.5 * nn) & (dists[:, 1:] < 1.8 * nn)).sum(axis=1)
    on_grid = n_nb >= 2
    if on_grid.sum() < 20:
        on_grid = n_nb >= 1
    return centers[on_grid]


def grid_transform(dots: np.ndarray, img_shape, dot_spacing_mm: float):
    """Affine transform that turns the image so the grid is axis-aligned and
    scales it to PX_PER_MM. Returns (M, out_w, out_h, spacing_px, angle_deg)."""
    if len(dots) < 5:
        raise ValueError(f"Only {len(dots)} grid dots found. Check the image "
                         "(focus, light, dots visible around the slab).")
    h, w = img_shape[:2]
    dists, idx = KDTree(dots).query(dots, k=5)
    spacing_px = np.median(dists[:, 1])

    # Dominant grid direction, modulo 90 degrees.
    angles = []
    for i in range(len(dots)):
        for j in range(1, 5):
            if 0.7 * spacing_px < dists[i, j] < 1.3 * spacing_px:
                d = dots[idx[i, j]] - dots[i]
                angles.append(np.arctan2(d[1], d[0]))
    hist, bins = np.histogram(np.array(angles) % (np.pi / 2), bins=180, range=(0, np.pi / 2))
    peak = np.argmax(gaussian_filter1d(hist.astype(float), sigma=3))
    grid_angle = np.degrees((bins[peak] + bins[peak + 1]) / 2)
    if grid_angle > 45:
        grid_angle -= 90

    scale = PX_PER_MM * dot_spacing_mm / spacing_px
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -grid_angle, scale)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    out_w = int(h * sin_a + w * cos_a)
    out_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (out_w - w) / 2
    M[1, 2] += (out_h - h) / 2
    return M, out_w, out_h, spacing_px, grid_angle


# ---------------------------------------------------------------------------
# Step 2: warped FOV outlines (BigWarp output)
# ---------------------------------------------------------------------------

def channel_to_binary(ch: np.ndarray, low: int, high: int) -> np.ndarray:
    ch = ch.astype(np.float64)
    if ch.max() > low:
        # Raw 16-bit fluorescence: clip to the bright-FOV window and stretch.
        c = np.clip(ch, low, high)
        return ((c - low) / (high - low) * 255).astype(np.uint8)
    # Outline export: shapes are bright lines. Binarize at half maximum.
    return (ch >= max(1.0, ch.max() * 0.5)).astype(np.uint8) * 255


def detect_trapezoid_and_square(img8: np.ndarray):
    """Find the trapezoid (largest quadrilateral) and the FOV square inside it."""
    edges = cv2.Canny(img8, 5, 50)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(cv2.dilate(edges, k, iterations=2), cv2.MORPH_CLOSE, k, iterations=3)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = closed.size * 0.001
    filtered = np.zeros_like(closed)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            filtered[labels == i] = 255

    contours, _ = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    trap, trap_hull = None, None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) > 5000:
            q, hull = find_quadrilateral(c)
            if q is not None and 4 <= len(q) <= 5:
                trap, trap_hull = q, hull
                break
    if trap is None:
        return None, None

    trap_area = cv2.contourArea(trap_hull)
    tcx, tcy = trap_hull[:, 0, 0].mean(), trap_hull[:, 0, 1].mean()
    trap_size = np.sqrt(trap_area)

    for edge_img in (filtered, closed):
        cs, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cs, key=cv2.contourArea, reverse=True)[:20]:
            a = cv2.contourArea(c)
            # Use the hull area for the upper limit: a dashed trapezoid outline
            # gives a thin ring contour with a small area, and it must not be
            # taken as the square.
            if a <= 200 or cv2.contourArea(cv2.convexHull(c)) >= trap_area * 0.5:
                continue
            rw, rh = cv2.minAreaRect(c)[1]
            if min(rw, rh) > 0 and max(rw, rh) / (min(rw, rh) + 1e-6) > 2.0:
                continue
            q, _ = find_quadrilateral(c)
            if q is None or not 4 <= len(q) <= 5:
                continue
            cx, cy = int(c[:, 0, 0].mean()), int(c[:, 0, 1].mean())
            inside = cv2.pointPolygonTest(trap_hull, (cx, cy), False) >= 0
            if not inside:
                inside = np.hypot(cx - tcx, cy - tcy) < trap_size * 1.5
            if inside and a < trap_area * 0.8:
                return trap, q
    return trap, None


def _intersect(p, r, q, s):
    """Intersection of the lines p + t*r and q + u*s (None if parallel)."""
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
    return p + t * r


def turn_edge(trap: np.ndarray, k: int, deg: float) -> np.ndarray:
    """Turn edge k (vertex k to k+1) by deg about its midpoint. The two new
    vertices stay on the lines of the two adjacent edges, so these edges keep
    their directions: parallel edges stay parallel and a 90 degree corner
    stays 90 degrees (the layout of the guide template)."""
    n = len(trap)
    v1, v2 = k, (k + 1) % n
    mid = (trap[v1] + trap[v2]) / 2
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    r = R @ (trap[v2] - trap[v1])
    out = trap.copy()
    a = _intersect(mid, r, trap[v1], trap[v1] - trap[(v1 - 1) % n])
    b = _intersect(mid, r, trap[v2], trap[(v2 + 1) % n] - trap[v2])
    if a is None or b is None:  # degenerate shape: turn the two vertices
        out[v1] = mid + R @ (trap[v1] - mid)
        out[v2] = mid + R @ (trap[v2] - mid)
    else:
        out[v1], out[v2] = a, b
    return out


def shape_trapezoid(trap: np.ndarray, square: np.ndarray | None, p: GuideParams):
    """Scale the trapezoid, turn its oblique edge, and make the cut lines."""
    trap = trap.copy()
    center = trap.mean(axis=0)
    trap = center + (trap - center) * np.sqrt(p.area_scale)

    n = len(trap)
    if abs(p.width_scale - 1.0) > 1e-6:
        # Scale along the axis between the midpoints of the two most-parallel edges.
        dirs = [(trap[(i + 1) % n] - trap[i]) / (np.linalg.norm(trap[(i + 1) % n] - trap[i]) + 1e-9)
                for i in range(n)]
        best, (i, j) = -1.0, (0, n // 2)
        for a in range(n):
            for b in range(a + 1, n):
                d = abs(np.dot(dirs[a], dirs[b]))
                if d > best:
                    best, (i, j) = d, (a, b)
        m1 = (trap[i] + trap[(i + 1) % n]) / 2
        m2 = (trap[j] + trap[(j + 1) % n]) / 2
        axis = (m2 - m1) / (np.linalg.norm(m2 - m1) + 1e-9)
        trap = trap + (p.width_scale - 1.0) * ((trap - center) @ axis)[:, None] * axis[None, :]

    # Edge most oblique to the square (largest minimum angle to any square edge).
    if square is not None:
        sq_dirs = [(square[(k + 1) % len(square)] - square[k]) /
                   np.linalg.norm(square[(k + 1) % len(square)] - square[k])
                   for k in range(len(square))]
    else:
        sq_dirs = [np.array([1, 0]), np.array([0, 1])]
    obliq = []
    for k in range(n):
        e = trap[(k + 1) % n] - trap[k]
        e = e / np.linalg.norm(e)
        ang = min(np.degrees(np.arccos(np.clip(abs(np.dot(e, s)), -1, 1))) for s in sq_dirs)
        obliq.append(ang)
    k = int(np.argmax(obliq))
    trap = turn_edge(trap, k, p.angle_increase_deg)

    cuts = []
    for k in range(n):
        a, b = trap[k], trap[(k + 1) % n]
        d = b - a
        cuts.append((a - d * p.edge_extend_factor, b + d * p.edge_extend_factor))
    return trap, cuts


def line_across_canvas(a: np.ndarray, b: np.ndarray, w: float, h: float):
    """Clip the infinite line through a and b to the canvas [0, w] x [0, h].
    Returns the two end points, or None if the line misses the canvas."""
    d = b - a
    t0, t1 = -np.inf, np.inf
    for p0, dp, lo, hi in ((a[0], d[0], 0.0, w), (a[1], d[1], 0.0, h)):
        if abs(dp) < 1e-12:
            if not lo <= p0 <= hi:
                return None
            continue
        ta, tb = sorted(((lo - p0) / dp, (hi - p0) / dp))
        t0, t1 = max(t0, ta), min(t1, tb)
    if t0 >= t1:
        return None
    return a + t0 * d, a + t1 * d


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_svg(boundary_px, fovs, dots_px, out_w, out_h, dot_spacing_mm,
              cut_color="black", cut_width_mm=0.176) -> str:
    b = boundary_px.reshape(-1, 2).astype(np.float64) / PX_PER_MM
    idx = np.linspace(0, len(b) - 1, min(200, len(b)), dtype=int)
    boundary_path = closed_bezier_path(b[idx])

    elems = ""
    for f in fovs:
        for i, (a, c) in enumerate(f.cut_lines):
            a, c = a / PX_PER_MM, c / PX_PER_MM
            elems += (f'  <line x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{c[0]:.3f}" y2="{c[1]:.3f}" '
                      f'stroke="{cut_color}" stroke-width="{cut_width_mm:g}" '
                      f'stroke-linecap="butt" id="ch{f.ch_idx}-trap-edge-{i}"/>\n')
        elems += (f'  <path d="{polygon_path(f.trapezoid / PX_PER_MM)}" fill="none" '
                  f'stroke="{f.color_name}" stroke-width="0.08" id="ch{f.ch_idx}-trapezoid-fov"/>\n')
        if f.square is not None:
            elems += (f'  <path d="{polygon_path(f.square / PX_PER_MM)}" fill="none" '
                      f'stroke="red" stroke-width="0.08" id="ch{f.ch_idx}-square-roi"/>\n')

    dots = "".join(f'    <circle cx="{x / PX_PER_MM:.3f}" cy="{y / PX_PER_MM:.3f}" r="0.1"/>\n'
                   for x, y in dots_px)
    W, H = out_w / PX_PER_MM, out_h / PX_PER_MM
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{W:.2f}mm" height="{H:.2f}mm" viewBox="0 0 {W:.2f} {H:.2f}">
  <title>Microdissection guide</title>
  <desc>Slab boundary, trapezoid cut outline(s), FOV square(s) and grid dots. 1 unit = 1 mm. Grid dot spacing = {dot_spacing_mm:g} mm. Made with fov_guide.py {__version__}.</desc>
  <path d="{boundary_path}" fill="none" stroke="black" stroke-width="0.1" id="brain-boundary"/>
{elems}  <g id="grid-dots" fill="gray" opacity="0.3">
{dots}  </g>
</svg>'''


def save_qc_plots(out_dir: Path, rgb, contour, boundary_t, fovs, dots_t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis = rgb.copy()
    cv2.drawContours(vis, [contour], -1, (255, 255, 0), 3)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(vis)
    ax.set_title("Automatic slab boundary")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "brain_boundary_overlay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    b = boundary_t.reshape(-1, 2).astype(np.float64)
    b = np.vstack([b, b[:1]])
    ax.plot(b[:, 0], b[:, 1], "k-", lw=2)
    for f in fovs:
        col = tuple(c / 255 for c in f.color_rgb)
        for a, c in f.cut_lines:
            ax.plot([a[0], c[0]], [a[1], c[1]], "-", color="k", lw=0.8)
        t = np.vstack([f.trapezoid, f.trapezoid[:1]])
        ax.plot(t[:, 0], t[:, 1], "-", color=col, lw=2, label=f"Ch {f.ch_idx} trapezoid")
        if f.square is not None:
            s = np.vstack([f.square, f.square[:1]])
            ax.plot(s[:, 0], s[:, 1], "r-", lw=2)
    ax.scatter(dots_t[:, 0], dots_t[:, 1], c="gray", s=10, alpha=0.4)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.set_title("Guide preview")
    fig.tight_layout()
    fig.savefig(out_dir / "svg_preview.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_fov_guide(image_path: str | Path, warped_tiff_path: str | Path,
                       output_svg_path: str | Path, params: GuideParams | None = None) -> Path:
    p = params or GuideParams()
    image_path, out_svg = Path(image_path), Path(output_svg_path)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1. Slab boundary and grid.
    slab_mask, contour = detect_slab_mask(rgb, p.brain_smooth_sigma)
    dots = detect_grid_dots(gray)
    print(f"  Grid dots found: {len(dots)}")
    M, out_w, out_h, spacing_px, grid_angle = grid_transform(dots, gray.shape, p.dot_spacing_mm)

    aligned = cv2.warpAffine(rgb, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                             borderValue=(255, 255, 255))
    aligned_path = out_svg.with_name(out_svg.stem + "_aligned.jpg")
    Image.fromarray(aligned).save(aligned_path, dpi=(ALIGNED_DPI, ALIGNED_DPI), quality=95)

    mask_t = cv2.warpAffine(slab_mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)
    cs, _ = cv2.findContours(mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary_t = smooth_contour(max(cs, key=cv2.contourArea), p.brain_smooth_sigma)
    dots_t = apply_affine(M, dots)

    # 2. Warped FOV outlines, one channel per FOV.
    warped = tifffile.imread(str(warped_tiff_path))
    if warped.ndim == 2:
        warped = warped[np.newaxis]
    print(f"  Warped TIFF channels: {warped.shape[0]}")

    fovs, binaries = [], []
    channels = p.channels if p.channels is not None else range(warped.shape[0])
    for ch in channels:
        if not 0 <= ch < warped.shape[0]:
            raise ValueError(f"Channel {ch} does not exist (TIFF has {warped.shape[0]}).")
        if warped[ch].max() == 0:
            print(f"    Channel {ch}: empty, skipped")
            continue
        filled = (warped[ch] > 0).mean()
        if filled > 0.05:
            print(f"    WARNING: channel {ch} has {filled:.0%} non-zero pixels. It looks like an "
                  f"image (for example the warped vasculature), not an outline. "
                  f"Use --channels to select the outline channels.")
        img8 = channel_to_binary(warped[ch], p.fluo_low, p.fluo_high)
        trap, sq = detect_trapezoid_and_square(img8)
        if trap is None:
            print(f"    Channel {ch}: no trapezoid found, skipped")
            continue
        name, rgb_c = FOV_COLORS[len(fovs) % len(FOV_COLORS)]
        trap_t = apply_affine(M, trap)
        sq_t = apply_affine(M, sq) if sq is not None else None
        trap_t, cuts = shape_trapezoid(trap_t, sq_t, p)
        if p.edge_extend_factor <= 0:
            cuts = [c for c in (line_across_canvas(trap_t[i], trap_t[(i + 1) % len(trap_t)], out_w, out_h)
                                for i in range(len(trap_t))) if c is not None]
        fovs.append(FOV(ch, trap_t, sq_t, cuts, name, rgb_c))
        binaries.append(img8)
        print(f"    Channel {ch}: trapezoid ({len(trap)} pts), "
              f"square ({'found' if sq is not None else 'NOT FOUND'})")
    if not fovs:
        raise ValueError("No trapezoid found in any channel of the warped TIFF.")

    # 3. Overlay image (QC and Inkscape reference).
    overlay = aligned.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for f, img8 in zip(fovs, binaries):
        m = cv2.dilate(cv2.warpAffine(img8, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
                                      borderValue=0), k, iterations=1) > 0
        overlay[m] = (0.5 * overlay[m].astype(np.float32)
                      + 0.5 * np.array(f.color_rgb, np.float32)).astype(np.uint8)
    Image.fromarray(overlay).save(out_svg.with_name(out_svg.stem + "_aligned_overlay.jpg"),
                                  dpi=(ALIGNED_DPI, ALIGNED_DPI), quality=95)

    # 4. SVG.
    out_svg.write_text(build_svg(boundary_t, fovs, dots_t, out_w, out_h, p.dot_spacing_mm,
                                 p.cut_color, p.cut_width_mm))
    if p.plot:
        save_qc_plots(out_svg.parent, rgb, contour, boundary_t, fovs, dots_t)

    print(f"SVG saved: {out_svg}")
    print(f"  Size: {out_w / PX_PER_MM:.1f} x {out_h / PX_PER_MM:.1f} mm")
    print(f"  Grid: {spacing_px:.1f} px = {p.dot_spacing_mm:g} mm, rotation {grid_angle:.2f} deg")
    print(f"  FOV channels: {[f.ch_idx for f in fovs]}")
    return out_svg


def find_inputs(folder: Path):
    jpgs = sorted(f for f in folder.glob("*.jpg") if "aligned" not in f.name.lower())
    tifs = sorted(folder.glob("*trap_warped.tif"))
    if not jpgs:
        raise SystemExit(f"No slab image (*.jpg) in {folder}")
    if not tifs:
        raise SystemExit(f"No *trap_warped.tif in {folder}")
    if len(jpgs) > 1 or len(tifs) > 1:
        print(f"  Note: more than one input found; using {jpgs[0].name} and {tifs[0].name}")
    return jpgs[0], tifs[0]


def pick_folder() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(title="Select the session folder")
        root.destroy()
        return Path(chosen) if chosen else None
    except Exception as e:  # no display
        print(f"  Folder dialog not available: {e}")
        return None


def main(argv=None):
    d = GuideParams()
    ap = argparse.ArgumentParser(description="Make a 1:1 SVG microdissection guide from a "
                                             "slab image and a BigWarp-warped FOV TIFF.")
    ap.add_argument("folder", nargs="?", help="Session folder (*.jpg and *trap_warped.tif). "
                                              "Omit to open a folder dialog.")
    ap.add_argument("--image", help="Slab image (overrides the folder search).")
    ap.add_argument("--warped", help="Warped FOV TIFF (overrides the folder search).")
    ap.add_argument("-o", "--output", help="Output SVG path.")
    ap.add_argument("--dot-spacing-mm", type=float, default=d.dot_spacing_mm,
                    help=f"Measured dot spacing of the printed guide, mm (default {d.dot_spacing_mm}).")
    ap.add_argument("--area-scale", type=float, default=d.area_scale)
    ap.add_argument("--width-scale", type=float, default=d.width_scale)
    ap.add_argument("--angle-increase-deg", type=float, default=d.angle_increase_deg)
    ap.add_argument("--edge-extend-factor", type=float, default=d.edge_extend_factor,
                    help="Cut line length as a multiple of the edge length. "
                         "0 (default) = lines across the whole guide.")
    ap.add_argument("--cut-color", default=d.cut_color, help="Cut line color (default black).")
    ap.add_argument("--cut-width-mm", type=float, default=d.cut_width_mm,
                    help="Cut line width in mm (default 0.176 = 0.5 pt).")
    ap.add_argument("--brain-smooth-sigma", type=float, default=d.brain_smooth_sigma)
    ap.add_argument("--channels", type=int, nargs="+",
                    help="0-based TIFF channels that hold FOV outlines (default: all).")
    ap.add_argument("--no-plot", action="store_true", help="Do not write QC images.")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    if a.image and a.warped:
        image, warped = Path(a.image), Path(a.warped)
        folder = image.parent
    else:
        folder = Path(a.folder) if a.folder else pick_folder()
        if folder is None or not folder.is_dir():
            raise SystemExit("No valid session folder.")
        image, warped = find_inputs(folder)
    out = Path(a.output) if a.output else folder / f"{folder.name}_fov_guide.svg"

    params = GuideParams(dot_spacing_mm=a.dot_spacing_mm, area_scale=a.area_scale,
                         width_scale=a.width_scale, angle_increase_deg=a.angle_increase_deg,
                         edge_extend_factor=a.edge_extend_factor, cut_color=a.cut_color,
                         cut_width_mm=a.cut_width_mm,
                         brain_smooth_sigma=a.brain_smooth_sigma, channels=a.channels,
                         plot=not a.no_plot)
    print(f"Session: {folder.name}\n  Image:  {image.name}\n  Warped: {warped.name}")
    generate_fov_guide(image, warped, out, params)


if __name__ == "__main__":
    main()
