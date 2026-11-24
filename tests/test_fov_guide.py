import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from fov_guide import GuideParams, generate_fov_guide  # noqa: E402
from make_synthetic_session import make_session  # noqa: E402


def test_synthetic_session(tmp_path):
    make_session(tmp_path)
    svg = generate_fov_guide(tmp_path / "slab.jpg", tmp_path / "slab_trap_warped.tif",
                             tmp_path / "guide.svg", GuideParams(dot_spacing_mm=0.93, plot=False))
    text = svg.read_text()
    assert 'id="brain-boundary"' in text
    assert 'id="ch0-trapezoid-fov"' in text
    assert 'id="ch0-square-roi"' in text
    assert len(re.findall(r'id="ch0-trap-edge-\d"', text)) == 4
    # 1 SVG unit = 1 mm: 2400 px image at 93 px/dot and 0.93 mm/dot is about 24 mm wide.
    w_mm = float(re.search(r'width="([\d.]+)mm"', text).group(1))
    assert 23 < w_mm < 27
    assert (tmp_path / "guide_aligned.jpg").exists()


def test_real_example(tmp_path):
    """Session 850399: channel 1 holds a dashed trapezoid and a small FOV square.
    The square must not be the trapezoid outline itself."""
    ex = ROOT / "examples" / "850399"
    svg = generate_fov_guide(ex / "IMG_20260901_153457_850399.jpg",
                             ex / "multiplane-ophys_850399_2026-07-28_10-32-20_trap_warped.tif",
                             tmp_path / "guide.svg",
                             GuideParams(dot_spacing_mm=0.93, channels=[1], plot=False))
    text = svg.read_text()

    def area(elem_id):
        d = re.search(r'd="([^"]+)"[^>]*id="%s"' % elem_id, text).group(1)
        p = [tuple(map(float, xy)) for xy in re.findall(r"([\d.]+),([\d.]+)", d)]
        return 0.5 * abs(sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p))))

    assert area("ch1-square-roi") < 0.25 * area("ch1-trapezoid-fov")
    assert "ch0-" not in text
