"""Coordinate transforms.

Every transform is a pure function over NumPy arrays.  Nothing here knows about
Plotly, Flask or JSON -- `figures.py` turns the results into traces.

Two families exist:

* ``curve``   -- an ordered sequence of 2D points, from ``y = f(x)`` or from a
  parametric pair ``(x(t), y(t))``.
* ``surface`` -- a grid of 3D points, from ``z = f(x, y)`` or from a parametric
  triple ``(x(u,v), y(u,v), z(u,v))``.

A transform *reinterprets* the same numbers in a different coordinate system,
which is the whole point of the app: ``polar`` reads a Cartesian pair as
(angle, radius) and shows where those numbers land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

CURVE = 'curve'
SURFACE = 'surface'
FIELD = 'field'

SQRT3_2 = float(np.sqrt(3.0) / 2.0)


@dataclass
class Mapped:
    """The result of applying a transform."""

    render: str                       # 'xy' | 'polar' | 'scene' | 'image'
    data: dict[str, np.ndarray]
    axes: tuple[str, ...] = ()
    #: 'force' -- the view is a geometric construction and must stay 1:1;
    #: 'source' -- follow the data (a parametric curve is isotropic, y = f(x)
    #: is not, because x and y carry unrelated units);
    #: 'free' -- the two coordinates are different quantities, let them scale.
    aspect: str = 'free'
    overlay: str | None = None
    colorbar: str | None = None
    note: str = ''


@dataclass(frozen=True)
class Transform:
    key: str
    label: str
    summary: str
    domain: str
    render: str
    apply: Callable[..., Mapped] = field(repr=False)
    equations: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# helpers


def _unwrap(angles: np.ndarray) -> np.ndarray:
    """Unwrap along the last axis, so surfaces do not tear at the +/-pi seam."""
    if angles.size == 0:
        return angles
    return np.unwrap(angles, axis=-1)


def _signed_polar(theta: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold negative radii onto the opposite ray, the usual polar convention."""
    negative = radius < 0
    theta = np.where(negative, theta + np.pi, theta)
    return np.degrees(np.mod(theta, 2 * np.pi)), np.abs(radius)


# ---------------------------------------------------------------------------
# curve transforms: (x, y) -> two coordinates


def _curve_cartesian(x: np.ndarray, y: np.ndarray) -> Mapped:
    return Mapped('xy', {'x': x, 'y': y}, ('x', 'y'), aspect='source')


def _curve_polar(x: np.ndarray, y: np.ndarray) -> Mapped:
    radius = np.hypot(x, y)
    theta = _unwrap(np.arctan2(y, x))
    return Mapped('xy', {'x': theta, 'y': radius}, ('θ (rad)', 'r'))


def _curve_polar_radial(x: np.ndarray, y: np.ndarray) -> Mapped:
    theta_deg, radius = _signed_polar(x, y)
    return Mapped(
        'polar', {'theta': theta_deg, 'r': radius}, ('θ', 'r'),
        note='x is read as the angle in radians, y as the radius.',
    )


def _curve_log_polar(x: np.ndarray, y: np.ndarray) -> Mapped:
    radius = np.hypot(x, y)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_radius = np.log(radius)
    theta = _unwrap(np.arctan2(y, x))
    return Mapped('xy', {'x': theta, 'y': log_radius}, ('θ (rad)', 'ln r'))


def _curve_triangular(x: np.ndarray, y: np.ndarray) -> Mapped:
    return Mapped(
        'xy', {'x': x - 0.5 * y, 'y': y * SQRT3_2}, ('', ''),
        aspect='force', overlay='triad2d',
        note='Axes 120 degrees apart, with z held at 0.',
    )


def _curve_hyperbolic(x: np.ndarray, y: np.ndarray) -> Mapped:
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.sqrt(np.abs(x * x - y * y))
        eta = 0.5 * np.log(np.abs((x + y) / (x - y)))
    return Mapped('xy', {'x': eta, 'y': rho}, ('η (rapidity)', 'ρ'))


def _curve_elliptic(x: np.ndarray, y: np.ndarray) -> Mapped:
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.arccosh((x + 1j * y).astype(np.complex128))
    return Mapped('xy', {'x': np.real(w), 'y': np.imag(w)}, ('μ', 'ν'))


def _curve_parabolic(x: np.ndarray, y: np.ndarray) -> Mapped:
    radius = np.hypot(x, y)
    with np.errstate(invalid='ignore'):
        tau = np.sqrt(np.maximum(radius + y, 0.0))
        sigma = np.sign(x) * np.sqrt(np.maximum(radius - y, 0.0))
    return Mapped('xy', {'x': sigma, 'y': tau}, ('σ', 'τ'))


# ---------------------------------------------------------------------------
# surface transforms: (X, Y, Z) -> two or three coordinates


def _surface_cartesian(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    return Mapped('scene', {'x': x, 'y': y, 'z': z}, ('x', 'y', 'z'), aspect='source')


def _surface_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    radius = np.sqrt(x * x + y * y + z * z)
    theta = _unwrap(np.arctan2(y, x))
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.arccos(np.clip(z / np.where(radius > 1e-12, radius, np.nan), -1.0, 1.0))
    return Mapped('scene', {'x': radius, 'y': theta, 'z': phi}, ('r', 'θ (rad)', 'φ (rad)'))


def _surface_cylindrical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    rho = np.hypot(x, y)
    phi = _unwrap(np.arctan2(y, x))
    return Mapped('scene', {'x': rho, 'y': phi, 'z': z}, ('ρ', 'φ (rad)', 'z'))


def _normalised(values: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(np.abs(values))) if values.size else 0.0
    return values / peak if np.isfinite(peak) and peak > 0 else values


def _surface_spherical_embed(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    """Read (x, y, z) as (theta, phi, r) and place the result back in space."""
    theta, phi = x, y
    radius = _normalised(np.abs(z))
    return Mapped(
        'scene',
        {
            'x': radius * np.sin(phi) * np.cos(theta),
            'y': radius * np.sin(phi) * np.sin(theta),
            'z': radius * np.cos(phi),
            'color': radius,
        },
        ('', '', ''),
        aspect='force',
        overlay='sphere-grid',
        colorbar='r',
        note='x is read as azimuth θ, y as polar angle φ, z as radius.',
    )


def _surface_cylindrical_embed(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    """Read (x, y, z) as (theta, rho, height): wraps the surface around an axis."""
    theta = x
    rho = 1.0 + _normalised(z)
    return Mapped(
        'scene',
        {'x': rho * np.cos(theta), 'y': rho * np.sin(theta), 'z': y, 'color': rho},
        ('', '', 'height'),
        aspect='force',
        colorbar='ρ',
        note='x is read as the azimuth, z modulates the radius, y becomes height.',
    )


def _surface_toroidal_embed(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    """Read (x, y, z) as (theta, phi, tube radius) on a torus."""
    theta, phi = x, y
    major, minor = 3.0, 1.0 + 0.6 * _normalised(z)
    return Mapped(
        'scene',
        {
            'x': (major + minor * np.cos(phi)) * np.cos(theta),
            'y': (major + minor * np.cos(phi)) * np.sin(theta),
            'z': minor * np.sin(phi),
            'color': minor,
        },
        ('', '', ''),
        aspect='force',
        colorbar='tube r',
        note='x and y are read as the two torus angles, z as the tube radius.',
    )


def _surface_triangular(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    """Project three axes onto one plane, 120 degrees apart."""
    return Mapped(
        'xy',
        {
            'x': x - 0.5 * y - 0.5 * z,
            'y': (y - z) * SQRT3_2,
            'color': z,
        },
        ('', ''),
        aspect='force',
        overlay='triad3d',
        colorbar='z',
        note='screen_x = x - y/2 - z/2,  screen_y = (y - z)·√3/2',
    )


def _surface_log_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Mapped:
    radius = np.sqrt(x * x + y * y + z * z)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_radius = np.log(radius)
        phi = np.arccos(np.clip(z / np.where(radius > 1e-12, radius, np.nan), -1.0, 1.0))
    theta = _unwrap(np.arctan2(y, x))
    return Mapped('scene', {'x': log_radius, 'y': theta, 'z': phi}, ('ln r', 'θ (rad)', 'φ (rad)'))


# ---------------------------------------------------------------------------
# field transforms: a complex map w = f(z) sampled on a grid


def _hsl_to_rgb(hue: np.ndarray, sat: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Vectorised HSL -> RGB, each channel in [0, 1]."""
    chroma = (1.0 - np.abs(2.0 * light - 1.0)) * sat
    sector = hue * 6.0
    second = chroma * (1.0 - np.abs(np.mod(sector, 2.0) - 1.0))
    zero = np.zeros_like(chroma)

    conditions = [sector < 1, sector < 2, sector < 3, sector < 4, sector < 5]
    red = np.select(conditions, [chroma, second, zero, zero, second], default=chroma)
    green = np.select(conditions, [second, chroma, chroma, second, zero], default=zero)
    blue = np.select(conditions, [zero, zero, second, chroma, chroma], default=second)

    offset = light - chroma / 2.0
    return np.clip(np.stack([red + offset, green + offset, blue + offset], axis=-1), 0.0, 1.0)


def _field_domain_coloring(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Mapped:
    """Hue carries the argument, lightness the modulus; zeros black, poles white."""
    magnitude = np.abs(w)
    finite = np.isfinite(magnitude)
    magnitude = np.where(finite, magnitude, np.inf)

    hue = np.mod(np.angle(np.where(finite, w, 1.0)) / (2.0 * np.pi), 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        light = 1.0 - 1.0 / (1.0 + magnitude ** 0.6)
        # Faint bands at every doubling of |w|, so level sets stay readable.
        bands = np.mod(np.log2(np.where(magnitude > 0, magnitude, np.nan)), 1.0)
    light = np.clip(np.nan_to_num(light, nan=0.0, posinf=1.0), 0.0, 1.0)
    light = np.clip(light + 0.06 * (np.nan_to_num(bands, nan=0.5) - 0.5), 0.0, 1.0)

    rgb = _hsl_to_rgb(hue, np.full_like(hue, 0.95), light)
    return Mapped(
        'image',
        {
            'rgb': (rgb * 255.0).astype(np.uint8),
            'x': np.array([x.min(), x.max()]),
            'y': np.array([y.min(), y.max()]),
        },
        ('Re z', 'Im z'),
        aspect='force',
        note='Hue = arg f(z), lightness = |f(z)|. Black is a zero, white is a pole.',
    )


def _field_conformal_grid(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Mapped:
    """The image of the Cartesian grid under f: rows and columns of the sample grid."""
    step = max(1, w.shape[0] // 28)
    rows, cols = w[::step, :], w[:, ::step].T
    return Mapped(
        'xy',
        {'rows': rows, 'cols': cols},
        ('Re f(z)', 'Im f(z)'),
        aspect='force',
        overlay='conformal',
        note='Horizontal and vertical lines of the z-plane, drawn where f sends them.',
    )


def _field_modulus_surface(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Mapped:
    magnitude = np.abs(w)
    return Mapped(
        'scene',
        {'x': x, 'y': y, 'z': magnitude, 'color': np.angle(w)},
        ('Re z', 'Im z', '|f(z)|'),
        colorbar='arg f',
        note='Height is the modulus, colour the argument.',
    )


def _field_phase_surface(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Mapped:
    return Mapped(
        'scene',
        {'x': x, 'y': y, 'z': np.angle(w), 'color': np.abs(w)},
        ('Re z', 'Im z', 'arg f(z)'),
        colorbar='|f|',
        note='Height is the argument, colour the modulus.',
    )


def _field_real_surface(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Mapped:
    return Mapped(
        'scene',
        {'x': x, 'y': y, 'z': np.real(w), 'color': np.imag(w)},
        ('Re z', 'Im z', 'Re f(z)'),
        colorbar='Im f',
    )


# ---------------------------------------------------------------------------
# registry

_CURVE_TRANSFORMS: tuple[Transform, ...] = (
    Transform('cartesian', 'Cartesian', 'The plane as sampled.',
              CURVE, 'xy', _curve_cartesian, ('x, y',)),
    Transform('polar', 'Polar (θ–r plane)', 'Radius against unwrapped angle.',
              CURVE, 'xy', _curve_polar, ('r = √(x²+y²)', 'θ = atan2(y, x)')),
    Transform('polar_radial', 'Polar (circular)', 'x read as angle, y as radius.',
              CURVE, 'polar', _curve_polar_radial, ('θ = x', 'r = y')),
    Transform('log_polar', 'Log-polar', 'Angle against log radius; scaling becomes a shift.',
              CURVE, 'xy', _curve_log_polar, ('ln r', 'θ')),
    Transform('triangular', 'Triangular (120°)', 'Oblique basis with three axes 120° apart.',
              CURVE, 'xy', _curve_triangular, ('u = x − y/2', 'v = y·√3/2')),
    Transform('hyperbolic', 'Hyperbolic', 'Minkowski rapidity and interval.',
              CURVE, 'xy', _curve_hyperbolic, ('ρ = √|x²−y²|', 'η = ½·ln|(x+y)/(x−y)|')),
    Transform('elliptic', 'Elliptic', 'Confocal ellipses and hyperbolae.',
              CURVE, 'xy', _curve_elliptic, ('μ + iν = arccosh(x + iy)',)),
    Transform('parabolic', 'Parabolic', 'Confocal parabolas.',
              CURVE, 'xy', _curve_parabolic, ('x = στ', 'y = (τ² − σ²)/2')),
)

_SURFACE_TRANSFORMS: tuple[Transform, ...] = (
    Transform('cartesian', 'Cartesian', 'The surface as sampled.',
              SURFACE, 'scene', _surface_cartesian, ('x, y, z',)),
    Transform('spherical', 'Spherical (r–θ–φ)', 'Spherical coordinates on Cartesian axes.',
              SURFACE, 'scene', _surface_spherical,
              ('r = √(x²+y²+z²)', 'θ = atan2(y, x)', 'φ = arccos(z/r)')),
    Transform('cylindrical', 'Cylindrical (ρ–φ–z)', 'Cylindrical coordinates on Cartesian axes.',
              SURFACE, 'scene', _surface_cylindrical,
              ('ρ = √(x²+y²)', 'φ = atan2(y, x)', 'z = z')),
    Transform('log_spherical', 'Log-spherical', 'Spherical with a logarithmic radius.',
              SURFACE, 'scene', _surface_log_spherical, ('ln r', 'θ', 'φ')),
    Transform('spherical_embed', 'Wrapped on a sphere', 'x → θ, y → φ, z → radius.',
              SURFACE, 'scene', _surface_spherical_embed,
              ('x = r·sinφ·cosθ', 'y = r·sinφ·sinθ', 'z = r·cosφ')),
    Transform('cylindrical_embed', 'Wrapped on a cylinder', 'x → angle, z → radius, y → height.',
              SURFACE, 'scene', _surface_cylindrical_embed,
              ('x = ρ·cosθ', 'y = ρ·sinθ', 'z = height')),
    Transform('toroidal_embed', 'Wrapped on a torus', 'x and y → torus angles, z → tube radius.',
              SURFACE, 'scene', _surface_toroidal_embed,
              ('x = (R + r·cosφ)·cosθ', 'y = (R + r·cosφ)·sinθ', 'z = r·sinφ')),
    Transform('triangular', 'Triangular (120°)', 'Three axes projected onto one plane.',
              SURFACE, 'xy', _surface_triangular,
              ('u = x − y/2 − z/2', 'v = (y − z)·√3/2')),
)

_FIELD_TRANSFORMS: tuple[Transform, ...] = (
    Transform('domain_coloring', 'Domain colouring', 'Hue is the argument, lightness the modulus.',
              FIELD, 'image', _field_domain_coloring,
              ('hue = arg f(z)', 'lightness = |f(z)|')),
    Transform('conformal_grid', 'Conformal grid', 'Where f sends the Cartesian grid.',
              FIELD, 'xy', _field_conformal_grid, ('f(x + iy)',)),
    Transform('modulus_surface', 'Modulus surface', 'Height |f|, colour arg f.',
              FIELD, 'scene', _field_modulus_surface, ('height = |f(z)|', 'colour = arg f(z)')),
    Transform('phase_surface', 'Phase surface', 'Height arg f, colour |f|.',
              FIELD, 'scene', _field_phase_surface, ('height = arg f(z)', 'colour = |f(z)|')),
    Transform('real_surface', 'Real part surface', 'Height Re f, colour Im f.',
              FIELD, 'scene', _field_real_surface, ('height = Re f(z)', 'colour = Im f(z)')),
)

_BY_DOMAIN: dict[str, tuple[Transform, ...]] = {
    CURVE: _CURVE_TRANSFORMS,
    SURFACE: _SURFACE_TRANSFORMS,
    FIELD: _FIELD_TRANSFORMS,
}

_INDEX: dict[tuple[str, str], Transform] = {
    (t.domain, t.key): t for group in _BY_DOMAIN.values() for t in group
}


def for_domain(domain: str) -> tuple[Transform, ...]:
    return _BY_DOMAIN.get(domain, ())


def get(domain: str, key: str) -> Transform:
    try:
        return _INDEX[(domain, key)]
    except KeyError:
        raise KeyError(f'unknown {domain} transform "{key}"') from None


def catalog() -> dict[str, list[dict[str, object]]]:
    """Serialisable description of every transform, for the client."""
    return {
        domain: [
            {
                'key': t.key,
                'label': t.label,
                'summary': t.summary,
                'render': t.render,
                'equations': list(t.equations),
            }
            for t in group
        ]
        for domain, group in _BY_DOMAIN.items()
    }


DEFAULT_VIEWS: dict[str, tuple[str, str, str]] = {
    CURVE: ('cartesian', 'polar', 'polar_radial'),
    SURFACE: ('cartesian', 'spherical', 'spherical_embed'),
    FIELD: ('domain_coloring', 'conformal_grid', 'modulus_surface'),
}
