"""Input sources: turn user expressions into sampled point sets.

Each source declares the expression fields, range fields and examples it needs.
The catalogue is served to the browser, so the form is built from this file
rather than being hard-coded twice.

Non-finite results are kept as NaN instead of being dropped.  Plotly breaks a
line and leaves a hole in a surface wherever it meets NaN, which is exactly how
a pole should look -- the previous behaviour of discarding every sample with
|x| < 0.05 punched a fake gap into perfectly well-behaved functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import transforms
from .expr import ExpressionError, as_array, evaluate, evaluate_scalar


@dataclass
class Sample:
    """Sampled data, ready for a transform."""

    domain: str
    coords: dict[str, np.ndarray]
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    caption: str = ''
    #: True when the coordinates carry the same units, so a 1:1 aspect is
    #: meaningful.  y = f(x) is not isotropic: x is a domain and y is a value.
    isotropic: bool = False


@dataclass(frozen=True)
class ExpressionField:
    name: str
    label: str
    placeholder: str
    default: str


@dataclass(frozen=True)
class RangeField:
    name: str
    label: str
    low: str
    high: str


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    summary: str
    domain: str
    variables: tuple[str, ...]
    expressions: tuple[ExpressionField, ...]
    ranges: tuple[RangeField, ...]
    samples: int
    samples_min: int
    samples_max: int
    samples_label: str
    build: Callable[..., Sample] = field(repr=False)
    examples: tuple[dict[str, Any], ...] = ()


def _finite(values: np.ndarray) -> np.ndarray:
    """Replace +/-inf with NaN so plots show a gap instead of a spike."""
    return np.where(np.isfinite(values), values, np.nan)


def _linspace(bounds: tuple[float, float], count: int) -> np.ndarray:
    low, high = bounds
    if low == high:
        raise ExpressionError('range bounds must differ')
    return np.linspace(low, high, count)


def _build_explicit_2d(exprs, bounds, samples, parameters) -> Sample:
    x = _linspace(bounds['x'], samples)
    y = as_array(evaluate(exprs['y'], {'x': x}, parameters), x)
    return Sample(transforms.CURVE, {'x': _finite(x), 'y': _finite(y)},
                  caption=f'y = {exprs["y"]}')


def _build_parametric_2d(exprs, bounds, samples, parameters) -> Sample:
    t = _linspace(bounds['t'], samples)
    x = as_array(evaluate(exprs['x'], {'t': t}, parameters), t)
    y = as_array(evaluate(exprs['y'], {'t': t}, parameters), t)
    return Sample(transforms.CURVE, {'x': _finite(x), 'y': _finite(y)},
                  caption=f'x = {exprs["x"]},  y = {exprs["y"]}', isotropic=True)


def _build_explicit_3d(exprs, bounds, samples, parameters) -> Sample:
    grid_x, grid_y = np.meshgrid(_linspace(bounds['x'], samples),
                                 _linspace(bounds['y'], samples))
    z = as_array(evaluate(exprs['z'], {'x': grid_x, 'y': grid_y}, parameters), grid_x)
    return Sample(transforms.SURFACE,
                  {'x': grid_x, 'y': grid_y, 'z': _finite(z)},
                  caption=f'z = {exprs["z"]}')


def _build_parametric_3d(exprs, bounds, samples, parameters) -> Sample:
    grid_u, grid_v = np.meshgrid(_linspace(bounds['u'], samples),
                                 _linspace(bounds['v'], samples))
    variables = {'u': grid_u, 'v': grid_v}
    coords = {
        axis: _finite(as_array(evaluate(exprs[axis], variables, parameters), grid_u))
        for axis in ('x', 'y', 'z')
    }
    caption = ',  '.join(f'{axis} = {exprs[axis]}' for axis in ('x', 'y', 'z'))
    return Sample(transforms.SURFACE, coords, caption=caption, isotropic=True)


def _build_complex(exprs, bounds, samples, parameters) -> Sample:
    grid_x, grid_y = np.meshgrid(_linspace(bounds['re'], samples),
                                 _linspace(bounds['im'], samples))
    z = grid_x + 1j * grid_y
    raw = evaluate(exprs['w'], {'z': z, 'x': grid_x, 'y': grid_y}, parameters)
    w = as_array(raw, grid_x, dtype=np.complex128)
    return Sample(transforms.FIELD,
                  {'x': grid_x, 'y': grid_y},
                  extras={'w': w},
                  caption=f'w = {exprs["w"]}', isotropic=True)


def _example(label: str, expressions: dict[str, str], **ranges: tuple[str, str]) -> dict[str, Any]:
    return {'label': label, 'expressions': expressions,
            'ranges': {k: {'low': v[0], 'high': v[1]} for k, v in ranges.items()}}


SOURCES: tuple[Source, ...] = (
    Source(
        key='explicit2d',
        label='Curve  y = f(x)',
        summary='A single-valued function of one variable.',
        domain=transforms.CURVE,
        variables=('x',),
        expressions=(ExpressionField('y', 'f(x)', 'sin(x)', 'sin(x)'),),
        ranges=(RangeField('x', 'x', '-2*pi', '2*pi'),),
        samples=1200, samples_min=50, samples_max=8000, samples_label='Samples',
        build=_build_explicit_2d,
        examples=(
            _example('sin x', {'y': 'sin(x)'}, x=('-2*pi', '2*pi')),
            _example('x²', {'y': 'x**2'}, x=('-3', '3')),
            _example('1/x', {'y': '1/x'}, x=('-3', '3')),
            _example('tan x', {'y': 'tan(x)'}, x=('-2*pi', '2*pi')),
            _example('Gaussian', {'y': 'exp(-x**2)'}, x=('-3', '3')),
            _example('Beat', {'y': 'sin(x)*cos(3*x)'}, x=('-2*pi', '2*pi')),
            _example('√|x|', {'y': 'sqrt(abs(x))'}, x=('-5', '5')),
            _example('Rose r(θ)', {'y': 'cos(5*x)'}, x=('0', '2*pi')),
        ),
    ),
    Source(
        key='parametric2d',
        label='Parametric curve  (x(t), y(t))',
        summary='Two functions of a shared parameter.',
        domain=transforms.CURVE,
        variables=('t',),
        expressions=(
            ExpressionField('x', 'x(t)', 'sin(3*t)', 'sin(3*t)'),
            ExpressionField('y', 'y(t)', 'sin(4*t)', 'sin(4*t)'),
        ),
        ranges=(RangeField('t', 't', '0', '2*pi'),),
        samples=2000, samples_min=50, samples_max=12000, samples_label='Samples',
        build=_build_parametric_2d,
        examples=(
            _example('Lissajous', {'x': 'sin(3*t)', 'y': 'sin(4*t)'}, t=('0', '2*pi')),
            _example('Rose', {'x': 'cos(5*t)*cos(t)', 'y': 'cos(5*t)*sin(t)'}, t=('0', '2*pi')),
            _example('Spiral', {'x': 't*cos(t)', 'y': 't*sin(t)'}, t=('0', '8*pi')),
            _example('Cycloid', {'x': 't-sin(t)', 'y': '1-cos(t)'}, t=('0', '6*pi')),
            _example('Cardioid', {'x': '(1-cos(t))*cos(t)', 'y': '(1-cos(t))*sin(t)'},
                     t=('0', '2*pi')),
            _example('Butterfly',
                     {'x': 'sin(t)*(exp(cos(t))-2*cos(4*t)-sin(t/12)**5)',
                      'y': 'cos(t)*(exp(cos(t))-2*cos(4*t)-sin(t/12)**5)'},
                     t=('0', '24*pi')),
        ),
    ),
    Source(
        key='explicit3d',
        label='Surface  z = f(x, y)',
        summary='A height field over the plane.',
        domain=transforms.SURFACE,
        variables=('x', 'y'),
        expressions=(ExpressionField('z', 'f(x, y)', 'sin(x)*cos(y)', 'sin(x)*cos(y)'),),
        ranges=(RangeField('x', 'x', '-3', '3'), RangeField('y', 'y', '-3', '3')),
        samples=72, samples_min=8, samples_max=240, samples_label='Grid resolution',
        build=_build_explicit_3d,
        examples=(
            _example('sin x · cos y', {'z': 'sin(x)*cos(y)'}, x=('-pi', 'pi'), y=('-pi', 'pi')),
            _example('Paraboloid', {'z': 'x**2 + y**2'}, x=('-3', '3'), y=('-3', '3')),
            _example('Saddle', {'z': 'x**2 - y**2'}, x=('-3', '3'), y=('-3', '3')),
            _example('Ripple', {'z': 'sin(sqrt(x**2 + y**2))'}, x=('-8', '8'), y=('-8', '8')),
            _example('Bell', {'z': 'exp(-(x**2 + y**2))'}, x=('-3', '3'), y=('-3', '3')),
            _example('Monkey saddle', {'z': 'x**3 - 3*x*y**2'}, x=('-2', '2'), y=('-2', '2')),
        ),
    ),
    Source(
        key='parametric3d',
        label='Parametric surface  (x, y, z)(u, v)',
        summary='Three functions of two parameters.',
        domain=transforms.SURFACE,
        variables=('u', 'v'),
        expressions=(
            ExpressionField('x', 'x(u, v)', 'sin(u)*cos(v)', 'sin(u)*cos(v)'),
            ExpressionField('y', 'y(u, v)', 'sin(u)*sin(v)', 'sin(u)*sin(v)'),
            ExpressionField('z', 'z(u, v)', 'cos(u)', 'cos(u)'),
        ),
        ranges=(RangeField('u', 'u', '0', 'pi'), RangeField('v', 'v', '0', '2*pi')),
        samples=72, samples_min=8, samples_max=240, samples_label='Grid resolution',
        build=_build_parametric_3d,
        examples=(
            _example('Sphere', {'x': 'sin(u)*cos(v)', 'y': 'sin(u)*sin(v)', 'z': 'cos(u)'},
                     u=('0', 'pi'), v=('0', '2*pi')),
            _example('Torus',
                     {'x': '(2+cos(u))*cos(v)', 'y': '(2+cos(u))*sin(v)', 'z': 'sin(u)'},
                     u=('0', '2*pi'), v=('0', '2*pi')),
            _example('Möbius band',
                     {'x': '(1+v/2*cos(u/2))*cos(u)',
                      'y': '(1+v/2*cos(u/2))*sin(u)',
                      'z': 'v/2*sin(u/2)'},
                     u=('0', '2*pi'), v=('-1', '1')),
            _example('Helicoid', {'x': 'v*cos(u)', 'y': 'v*sin(u)', 'z': 'u/2'},
                     u=('0', '4*pi'), v=('-2', '2')),
            _example('Seashell',
                     {'x': '(1-u/(2*pi))*cos(3*u)*(1+cos(v)) + 0.2*cos(3*u)',
                      'y': '(1-u/(2*pi))*sin(3*u)*(1+cos(v)) + 0.2*sin(3*u)',
                      'z': 'u/pi + (1-u/(2*pi))*sin(v)'},
                     u=('0', '2*pi'), v=('0', '2*pi')),
            _example('Klein bottle (figure 8)',
                     {'x': '(2+cos(u/2)*sin(v)-sin(u/2)*sin(2*v))*cos(u)',
                      'y': '(2+cos(u/2)*sin(v)-sin(u/2)*sin(2*v))*sin(u)',
                      'z': 'sin(u/2)*sin(v)+cos(u/2)*sin(2*v)'},
                     u=('0', '2*pi'), v=('0', '2*pi')),
        ),
    ),
    Source(
        key='complex',
        label='Complex map  w = f(z)',
        summary='A function of a complex variable, sampled on the z-plane.',
        domain=transforms.FIELD,
        variables=('z', 'x', 'y'),
        expressions=(ExpressionField('w', 'f(z)', 'z**2', 'z**2'),),
        ranges=(RangeField('re', 'Re z', '-2', '2'), RangeField('im', 'Im z', '-2', '2')),
        samples=300, samples_min=40, samples_max=700, samples_label='Grid resolution',
        build=_build_complex,
        examples=(
            _example('z²', {'w': 'z**2'}, re=('-2', '2'), im=('-2', '2')),
            _example('1/z', {'w': '1/z'}, re=('-2', '2'), im=('-2', '2')),
            _example('Möbius', {'w': '(z-1)/(z+1)'}, re=('-3', '3'), im=('-3', '3')),
            _example('Roots of unity', {'w': 'z**3 - 1'}, re=('-2', '2'), im=('-2', '2')),
            _example('exp z', {'w': 'exp(z)'}, re=('-3', '3'), im=('-pi', 'pi')),
            _example('sin z', {'w': 'sin(z)'}, re=('-pi', 'pi'), im=('-2', '2')),
            _example('Blaschke', {'w': '(z**2-0.25)/(1-0.25*z**2)'}, re=('-2', '2'), im=('-2', '2')),
        ),
    ),
)

_BY_KEY = {source.key: source for source in SOURCES}


def get(key: str) -> Source:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f'unknown source "{key}"') from None


def sample(
    source: Source,
    expressions: Mapping[str, str],
    ranges: Mapping[str, Mapping[str, Any]],
    samples: int,
    parameters: Mapping[str, float],
) -> Sample:
    """Validate the request against `source` and produce the sampled data."""
    missing = [f.name for f in source.expressions if not str(expressions.get(f.name, '')).strip()]
    if missing:
        raise ExpressionError(f'missing expression for {", ".join(missing)}')

    bounds: dict[str, tuple[float, float]] = {}
    for spec in source.ranges:
        given = ranges.get(spec.name) or {}
        low = evaluate_scalar(given.get('low', spec.low), parameters)
        high = evaluate_scalar(given.get('high', spec.high), parameters)
        if low == high:
            raise ExpressionError(f'{spec.label} range is empty')
        bounds[spec.name] = (min(low, high), max(low, high))

    count = int(np.clip(int(samples or source.samples), source.samples_min, source.samples_max))
    values = {f.name: str(expressions[f.name]).strip() for f in source.expressions}
    return source.build(values, bounds, count, dict(parameters))


def catalog() -> list[dict[str, Any]]:
    """Serialisable description of every source, for the client."""
    return [
        {
            'key': source.key,
            'label': source.label,
            'summary': source.summary,
            'domain': source.domain,
            'variables': list(source.variables),
            'expressions': [
                {'name': f.name, 'label': f.label,
                 'placeholder': f.placeholder, 'default': f.default}
                for f in source.expressions
            ],
            'ranges': [
                {'name': r.name, 'label': r.label, 'low': r.low, 'high': r.high}
                for r in source.ranges
            ],
            'samples': {
                'default': source.samples, 'min': source.samples_min,
                'max': source.samples_max, 'label': source.samples_label,
            },
            'examples': list(source.examples),
            'defaultViews': list(transforms.DEFAULT_VIEWS[source.domain]),
        }
        for source in SOURCES
    ]


def default_source() -> Source:
    return SOURCES[0]


def example_labels(source: Source) -> Sequence[str]:
    return [example['label'] for example in source.examples]
