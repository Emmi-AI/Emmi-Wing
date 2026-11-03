import numpy as np
import pytest

from generate_wing_mesh import generate_wing_mesh, naca4_airfoil_points, parse_naca4


@pytest.mark.parametrize(
    "digits", ["0012", "NACA 0012", 12, ("0", "0", "1", "2"), (0, 0, 12)]
)
def test_naca0012_is_twelve_percent_thick(digits):
    _, z = naca4_airfoil_points(digits, n_points=400)
    assert z.max() - z.min() == pytest.approx(0.12, abs=2e-3)
    assert z.max() == pytest.approx(-z.min())  # symmetric


def test_parse_naca4():
    assert parse_naca4("2412") == pytest.approx((0.02, 0.4, 0.12))
    assert parse_naca4((2, 4, 12)) == pytest.approx((0.02, 0.4, 0.12))
    with pytest.raises(ValueError):
        parse_naca4("241")


def test_wing_thickness_scales_with_chord():
    wing = generate_wing_mesh(
        "0012",
        chord_root=2.0,
        span=4.0,
        taper_ratio=0.5,
        sweep=0.0,
        n_chord=200,
        n_span=20,
    )
    pts = wing.vectors.reshape(-1, 3)
    root = pts[np.isclose(pts[:, 1], 0.0)]
    assert root[:, 2].max() - root[:, 2].min() == pytest.approx(0.24, abs=5e-3)
