"""The transforms are pure maths, so they can be checked against closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from dimensionmorph import transforms
from dimensionmorph.figures import _png_data_uri, _polyline, _robust_range

ANGLES = np.linspace(0.2, 2.0, 40)


def test_every_registered_transform_is_reachable():
    for domain, group in transforms.catalog().items():
        for entry in group:
            assert transforms.get(domain, str(entry['key'])).label == entry['label']


def test_unknown_transform_raises():
    with pytest.raises(KeyError):
        transforms.get('curve', 'nope')


def test_default_views_exist_for_every_domain():
    for domain, keys in transforms.DEFAULT_VIEWS.items():
        for key in keys:
            assert transforms.get(domain, key)


def test_polar_inverts_to_the_original_point():
    x, y = np.cos(ANGLES) * 2.0, np.sin(ANGLES) * 2.0
    mapped = transforms.get('curve', 'polar').apply(x, y)
    theta, radius = mapped.data['x'], mapped.data['y']
    np.testing.assert_allclose(radius * np.cos(theta), x, atol=1e-9)
    np.testing.assert_allclose(radius * np.sin(theta), y, atol=1e-9)


def test_polar_angle_is_unwrapped():
    """Crossing +/-pi must not leave a 2*pi jump for a surface to tear on."""
    theta = np.linspace(np.pi - 0.3, np.pi + 0.3, 50)
    mapped = transforms.get('curve', 'polar').apply(np.cos(theta), np.sin(theta))
    assert np.max(np.abs(np.diff(mapped.data['x']))) < 0.1


def test_polar_radial_folds_negative_radii():
    theta = np.array([0.0, 0.0])
    radius = np.array([1.0, -1.0])
    mapped = transforms.get('curve', 'polar_radial').apply(theta, radius)
    np.testing.assert_allclose(mapped.data['r'], [1.0, 1.0])
    np.testing.assert_allclose(mapped.data['theta'], [0.0, 180.0])


def test_triangular_axes_are_120_degrees_apart():
    apply = transforms.get('curve', 'triangular').apply
    unit_x = apply(np.array([1.0]), np.array([0.0]))
    unit_y = apply(np.array([0.0]), np.array([1.0]))
    direction_x = np.array([unit_x.data['x'][0], unit_x.data['y'][0]])
    direction_y = np.array([unit_y.data['x'][0], unit_y.data['y'][0]])
    cosine = direction_x @ direction_y / (np.linalg.norm(direction_x) * np.linalg.norm(direction_y))
    assert np.degrees(np.arccos(cosine)) == pytest.approx(120.0)


def test_surface_triangular_axes_are_120_degrees_apart():
    apply = transforms.get('surface', 'triangular').apply
    one, zero = np.array([1.0]), np.array([0.0])
    directions = [
        np.array([m.data['x'][0], m.data['y'][0]])
        for m in (apply(one, zero, zero), apply(zero, one, zero), apply(zero, zero, one))
    ]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cosine = directions[a] @ directions[b] / (
            np.linalg.norm(directions[a]) * np.linalg.norm(directions[b]))
        assert np.degrees(np.arccos(cosine)) == pytest.approx(120.0)


def test_spherical_inverts_to_the_original_point():
    grid = np.linspace(0.5, 2.0, 12)
    x, y = np.meshgrid(grid, grid)
    z = 0.5 * x + 0.25 * y
    mapped = transforms.get('surface', 'spherical').apply(x, y, z)
    radius, theta, phi = mapped.data['x'], mapped.data['y'], mapped.data['z']
    np.testing.assert_allclose(radius * np.sin(phi) * np.cos(theta), x, atol=1e-9)
    np.testing.assert_allclose(radius * np.sin(phi) * np.sin(theta), y, atol=1e-9)
    np.testing.assert_allclose(radius * np.cos(phi), z, atol=1e-9)


def test_cylindrical_inverts_to_the_original_point():
    grid = np.linspace(0.5, 2.0, 12)
    x, y = np.meshgrid(grid, grid)
    z = x * 0.5
    mapped = transforms.get('surface', 'cylindrical').apply(x, y, z)
    rho, phi = mapped.data['x'], mapped.data['y']
    np.testing.assert_allclose(rho * np.cos(phi), x, atol=1e-9)
    np.testing.assert_allclose(rho * np.sin(phi), y, atol=1e-9)
    np.testing.assert_allclose(mapped.data['z'], z)


def test_parabolic_inverts_to_the_original_point():
    grid = np.linspace(-2.0, 2.0, 17)
    x, y = grid, grid * 0.5 + 1.0
    mapped = transforms.get('curve', 'parabolic').apply(x, y)
    sigma, tau = mapped.data['x'], mapped.data['y']
    np.testing.assert_allclose(sigma * tau, x, atol=1e-9)
    np.testing.assert_allclose((tau ** 2 - sigma ** 2) / 2, y, atol=1e-9)


def test_elliptic_inverts_to_the_original_point():
    grid = np.linspace(0.4, 1.8, 15)
    x, y = grid, grid * 0.7
    mapped = transforms.get('curve', 'elliptic').apply(x, y)
    w = mapped.data['x'] + 1j * mapped.data['y']
    np.testing.assert_allclose(np.real(np.cosh(w)), x, atol=1e-9)
    np.testing.assert_allclose(np.imag(np.cosh(w)), y, atol=1e-9)


def test_hyperbolic_recovers_the_interval():
    x = np.linspace(1.5, 3.0, 20)
    y = x * 0.4
    mapped = transforms.get('curve', 'hyperbolic').apply(x, y)
    np.testing.assert_allclose(mapped.data['y'], np.sqrt(x ** 2 - y ** 2), atol=1e-9)
    np.testing.assert_allclose(mapped.data['x'], np.arctanh(y / x), atol=1e-9)


def test_domain_colouring_marks_zeros_black_and_poles_white():
    grid = np.array([[0.0, 1.0e9]])
    mapped = transforms.get('field', 'domain_coloring').apply(grid, grid, grid.astype(complex))
    rgb = mapped.data['rgb']
    assert rgb[0, 0].max() < 25, 'a zero should be black'
    assert rgb[0, 1].min() > 230, 'a pole should be white'


def test_conformal_grid_returns_both_line_families():
    grid = np.linspace(-1.0, 1.0, 24)
    x, y = np.meshgrid(grid, grid)
    mapped = transforms.get('field', 'conformal_grid').apply(x, y, (x + 1j * y) ** 2)
    assert mapped.data['rows'].size and mapped.data['cols'].size


# --------------------------------------------------------------- figures helpers


def test_polyline_separates_segments_with_nan():
    xs, ys = _polyline([(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                        (np.array([2.0, 3.0]), np.array([2.0, 3.0]))])
    assert np.isnan(xs[2]) and np.isnan(ys[2])
    assert xs[3] == 2.0


def test_robust_range_clips_a_pole_but_keeps_tame_data():
    tame = np.linspace(-1.0, 1.0, 500)
    tame_range = _robust_range(tame)
    assert tame_range is not None
    low, high = tame_range
    assert low < -1.0 and high > 1.0 and high < 1.5

    spiky = np.concatenate([tame, np.array([1e6])])
    spiky_range = _robust_range(spiky)
    assert spiky_range is not None
    _, high = spiky_range
    assert high < 100.0, 'a single spike should not flatten the rest'


def test_robust_range_handles_all_nan():
    assert _robust_range(np.full(10, np.nan)) is None


def test_png_encoder_produces_a_valid_png_header():
    rgb = np.zeros((4, 3, 3), dtype=np.uint8)
    uri = _png_data_uri(rgb)
    assert uri.startswith('data:image/png;base64,')

    import base64
    raw = base64.b64decode(uri.split(',', 1)[1])
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'
    width = int.from_bytes(raw[16:20], 'big')
    height = int.from_bytes(raw[20:24], 'big')
    assert (width, height) == (3, 4)
