"""Build Plotly figure dictionaries from sampled data.

Two things matter here.

*Payload size.*  Reference grids are emitted as a single polyline trace with
NaN separators rather than one trace per line.  The 3D tessellation view used
to ship 196 traces and 454 KB; the same picture is now 4 traces and a fraction
of the bytes.

*JSON validity.*  NumPy NaN serialises as the bare token ``NaN``, which
``JSON.parse`` rejects, so every array goes through :func:`_to_list`, which
turns non-finite values into ``null``.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, Iterable, Sequence

import numpy as np

from .sources import Sample
from .transforms import Mapped, Transform

SQRT3_2 = float(np.sqrt(3.0) / 2.0)

#: Maximum grid side fed to a 3D surface trace; complex fields are sampled far
#: more finely than a scene can usefully display.
MAX_SCENE_SIDE = 120
#: Above this many points a curve is drawn with WebGL instead of SVG.
WEBGL_THRESHOLD = 4000
#: Significant digits kept in the JSON payload.  Seven is well past what a
#: screen can show, and dropping the rest roughly halves the response.
SIGNIFICANT_DIGITS = 7

THEMES: dict[str, dict[str, Any]] = {
    'light': {
        'paper': '#ffffff', 'plot': '#ffffff',
        'grid': '#e7e9f0', 'zero': '#c3c8d6', 'line': '#d7dae4',
        'text': '#2f323b', 'muted': '#767c8c',
        'accent': '#4f5ce8', 'accent2': '#d1495b', 'accent3': '#1b998b',
        'axis_x': '#d1495b', 'axis_y': '#2f6fed', 'axis_z': '#149a72',
        'overlay': 'rgba(120,128,150,0.30)',
        'surface': 'Viridis',
    },
    'dark': {
        'paper': '#161a21', 'plot': '#161a21',
        'grid': '#282e39', 'zero': '#3b4353', 'line': '#333a47',
        'text': '#e2e6ef', 'muted': '#9aa2b4',
        'accent': '#8b96ff', 'accent2': '#ff8a9b', 'accent3': '#4fd1c5',
        'axis_x': '#ff7d86', 'axis_y': '#6fa8ff', 'axis_z': '#4fd1a0',
        'overlay': 'rgba(160,170,196,0.26)',
        'surface': 'Viridis',
    },
}

#: Cyclic scale for phase, so arg = -pi and arg = +pi meet without a seam.
CYCLIC_SCALE = [
    [0.0, '#e6e1f5'], [0.125, '#9a8fe0'], [0.25, '#4c6edb'], [0.375, '#2f9bb5'],
    [0.5, '#3fb37a'], [0.625, '#9bb03f'], [0.75, '#d98032'], [0.875, '#cf4a6e'],
    [1.0, '#e6e1f5'],
]


def theme(name: str) -> dict[str, Any]:
    return THEMES.get(name, THEMES['light'])


# ---------------------------------------------------------------------------
# serialisation helpers


def _to_list(values: Any) -> list:
    """NumPy array -> JSON-safe nested list, with non-finite entries as null."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.tolist()

    finite = np.isfinite(array)
    if finite.any():
        # Round to a fixed number of significant digits.  Python writes the
        # shortest round-trip repr of a float, so this genuinely shrinks the
        # response rather than only hiding digits.
        peak = float(np.max(np.abs(array[finite])))
        if peak > 0:
            exponent = int(np.floor(np.log10(peak)))
            array = np.round(array, max(0, SIGNIFICANT_DIGITS - 1 - exponent))

    if not finite.all():
        boxed = array.astype(object)
        boxed[~finite] = None
        return boxed.tolist()
    return array.tolist()


def _png_data_uri(rgb: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 array as a PNG data URI using only stdlib."""
    height, width = rgb.shape[:2]
    contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)
    # One filter byte (0 = None) in front of every scanline.
    scanlines = np.hstack([np.zeros((height, 1), np.uint8), contiguous.reshape(height, -1)])

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack('>I', len(payload)) + body + struct.pack('>I', zlib.crc32(body))

    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    data = (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', header) +
            chunk(b'IDAT', zlib.compress(scanlines.tobytes(), 6)) +
            chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def _polyline(segments: Iterable[Sequence[np.ndarray]]) -> tuple[np.ndarray, ...]:
    """Join many line segments into one NaN-separated polyline per axis."""
    collected: list[list[np.ndarray]] = []
    for segment in segments:
        if not collected:
            collected = [[] for _ in segment]
        for axis, values in enumerate(segment):
            collected[axis].append(np.asarray(values, dtype=np.float64))
            collected[axis].append(np.array([np.nan]))
    if not collected:
        return ()
    return tuple(np.concatenate(parts) if parts else np.array([]) for parts in collected)


def _decimate(*arrays: np.ndarray, limit: int = MAX_SCENE_SIDE) -> tuple[np.ndarray, ...]:
    """Thin a 2D grid so a scene trace stays responsive."""
    first = arrays[0]
    if first.ndim != 2:
        return arrays
    step = max(1, int(np.ceil(max(first.shape) / limit)))
    if step == 1:
        return arrays
    return tuple(a[::step, ::step] for a in arrays)


# ---------------------------------------------------------------------------
# axis ranges


def _robust_range(values: np.ndarray, pad: float = 0.06) -> list[float] | None:
    """Data bounds, clipped back to the bulk of the data when a pole blows up."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    low, high = float(finite.min()), float(finite.max())
    q_low, q_high = (float(v) for v in np.percentile(finite, [1.0, 99.0]))
    inner = q_high - q_low
    if inner > 0 and (high - low) > 12.0 * inner:
        low, high = q_low - 0.5 * inner, q_high + 0.5 * inner

    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0)
        low, high = low - span, high + span
    return [low - span * pad, high + span * pad]


def _match_aspect(x_range: list[float] | None,
                  y_range: list[float] | None) -> tuple[list[float] | None, list[float] | None]:
    """Widen the shorter axis so a 1:1 aspect does not crop the data."""
    if not x_range or not y_range:
        return x_range, y_range
    span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])

    def centred(rng: list[float]) -> list[float]:
        mid = 0.5 * (rng[0] + rng[1])
        return [mid - span / 2.0, mid + span / 2.0]

    return centred(x_range), centred(y_range)


# ---------------------------------------------------------------------------
# layout


def _base_layout(palette: dict[str, Any]) -> dict[str, Any]:
    """Chrome-free layout: the panel header already names the view."""
    return {
        'paper_bgcolor': palette['paper'],
        'plot_bgcolor': palette['plot'],
        'font': {'family': 'Inter, "Segoe UI", system-ui, sans-serif',
                 'size': 12, 'color': palette['text']},
        'margin': {'l': 54, 'r': 18, 't': 14, 'b': 42},
        'showlegend': False,
        'hovermode': 'closest',
        'hoverlabel': {'bgcolor': palette['paper'], 'bordercolor': palette['line'],
                       'font': {'color': palette['text'], 'size': 11}},
        'autosize': True,
        'dragmode': 'pan',          # flat plots; scenes override this below
    }


def _axis(palette: dict[str, Any], title: str, rng: list[float] | None,
          visible: bool = True) -> dict[str, Any]:
    axis: dict[str, Any] = {
        'title': {'text': title, 'font': {'size': 11, 'color': palette['muted']}},
        'gridcolor': palette['grid'],
        'zerolinecolor': palette['zero'],
        'linecolor': palette['line'],
        'tickfont': {'size': 10, 'color': palette['muted']},
        'zeroline': True,
        'showgrid': visible,
        'showticklabels': visible,
    }
    if rng:
        axis['range'] = rng
    return axis


def _scene_axis(palette: dict[str, Any], title: str, visible: bool = True) -> dict[str, Any]:
    return {
        'title': {'text': title, 'font': {'size': 11, 'color': palette['muted']}},
        'gridcolor': palette['grid'],
        'zerolinecolor': palette['zero'],
        'backgroundcolor': palette['plot'],
        'showbackground': True,
        'tickfont': {'size': 9, 'color': palette['muted']},
        'showgrid': visible,
        'showticklabels': visible,
        'visible': visible,
    }


# ---------------------------------------------------------------------------
# overlays


def _triangular_grid(extent: float, palette: dict[str, Any], divisions: int = 9) -> dict[str, Any]:
    """One trace holding a triangular lattice: lines along three 120° directions."""
    directions = [(1.0, 0.0), (-0.5, SQRT3_2), (-0.5, -SQRT3_2)]
    offsets = np.linspace(-extent, extent, 2 * divisions + 1)
    reach = extent * 2.2
    segments = []
    for dx, dy in directions:
        nx, ny = -dy, dx  # normal to this direction
        for offset in offsets:
            base_x, base_y = nx * offset, ny * offset
            segments.append((
                np.array([base_x - dx * reach, base_x + dx * reach]),
                np.array([base_y - dy * reach, base_y + dy * reach]),
            ))
    grid_x, grid_y = _polyline(segments)
    return {
        'type': 'scattergl', 'mode': 'lines',
        'x': _to_list(grid_x), 'y': _to_list(grid_y),
        'line': {'color': palette['overlay'], 'width': 1},
        'hoverinfo': 'skip', 'showlegend': False,
    }


def _triad_traces(extent: float, palette: dict[str, Any], labels: Sequence[str]) -> list[dict]:
    """The three oblique axes plus their labels: two traces, not eight."""
    directions = [(1.0, 0.0), (-0.5, SQRT3_2), (-0.5, -SQRT3_2)][:len(labels)]
    colours = [palette['axis_x'], palette['axis_y'], palette['axis_z']][:len(labels)]

    traces = []
    for (dx, dy), colour in zip(directions, colours):
        traces.append({
            'type': 'scatter', 'mode': 'lines',
            'x': [0.0, dx * extent], 'y': [0.0, dy * extent],
            'line': {'color': colour, 'width': 2},
            'hoverinfo': 'skip', 'showlegend': False,
        })
    traces.append({
        'type': 'scatter', 'mode': 'text',
        'x': [dx * extent * 1.08 for dx, _ in directions],
        'y': [dy * extent * 1.08 for _, dy in directions],
        'text': list(labels),
        'textfont': {'size': 13, 'color': palette['text']},
        'hoverinfo': 'skip', 'showlegend': False,
    })
    return traces


def _sphere_grid(palette: dict[str, Any], radius: float = 1.0) -> dict[str, Any]:
    """Meridians and parallels as a single 3D polyline."""
    fine = np.linspace(0.0, 2.0 * np.pi, 120)
    segments = []
    for phi in np.linspace(np.pi / 8, np.pi * 7 / 8, 7):          # parallels
        ring = radius * np.sin(phi)
        segments.append((ring * np.cos(fine), ring * np.sin(fine),
                         np.full_like(fine, radius * np.cos(phi))))
    arc = np.linspace(0.0, np.pi, 60)
    for theta in np.linspace(0.0, np.pi, 6, endpoint=False):      # meridians
        segments.append((radius * np.sin(arc) * np.cos(theta),
                         radius * np.sin(arc) * np.sin(theta),
                         radius * np.cos(arc)))
    grid_x, grid_y, grid_z = _polyline(segments)
    return {
        'type': 'scatter3d', 'mode': 'lines',
        'x': _to_list(grid_x), 'y': _to_list(grid_y), 'z': _to_list(grid_z),
        'line': {'color': palette['overlay'], 'width': 1},
        'hoverinfo': 'skip', 'showlegend': False,
    }


# ---------------------------------------------------------------------------
# renderers


def _render_xy_curve(mapped: Mapped, palette: dict[str, Any]) -> tuple[list[dict], dict]:
    x, y = mapped.data['x'], mapped.data['y']
    trace_type = 'scattergl' if x.size > WEBGL_THRESHOLD else 'scatter'
    traces: list[dict] = []

    square = mapped.aspect == 'force'
    x_range, y_range = _robust_range(x), _robust_range(y)
    if square:
        x_range, y_range = _match_aspect(x_range, y_range)

    if mapped.overlay == 'triad2d':
        extent = max(abs(v) for v in (x_range or [1]) + (y_range or [1])) or 1.0
        traces.append(_triangular_grid(extent, palette))
        traces.extend(_triad_traces(extent * 0.82, palette, ('X', 'Y', 'Z')))

    traces.append({
        'type': trace_type, 'mode': 'lines',
        'x': _to_list(x), 'y': _to_list(y),
        'line': {'color': palette['accent'], 'width': 2.2, 'shape': 'linear'},
        'hovertemplate': '%{x:.4g}, %{y:.4g}<extra></extra>',
        'showlegend': False,
    })

    axes = mapped.axes or ('', '')
    layout: dict[str, Any] = {
        'xaxis': _axis(palette, axes[0], x_range),
        'yaxis': _axis(palette, axes[1], y_range),
    }
    if square:
        layout['yaxis']['scaleanchor'] = 'x'
        layout['yaxis']['scaleratio'] = 1
    if mapped.overlay == 'triad2d':
        layout['xaxis']['showgrid'] = layout['yaxis']['showgrid'] = False
        layout['xaxis']['zeroline'] = layout['yaxis']['zeroline'] = False
    return traces, layout


def _render_xy_scatter(mapped: Mapped, palette: dict[str, Any]) -> tuple[list[dict], dict]:
    """A projected surface: coloured points in the plane."""
    x, y = mapped.data['x'].ravel(), mapped.data['y'].ravel()
    colour = mapped.data.get('color')
    x_range, y_range = _match_aspect(_robust_range(x), _robust_range(y))

    traces: list[dict] = []
    if mapped.overlay == 'triad3d':
        extent = max(abs(v) for v in (x_range or [1]) + (y_range or [1])) or 1.0
        traces.append(_triangular_grid(extent, palette))
        traces.extend(_triad_traces(extent * 0.82, palette, ('X', 'Y', 'Z')))

    marker: dict[str, Any] = {'size': 3.2, 'opacity': 0.85}
    if colour is not None:
        marker.update({
            'color': _to_list(colour.ravel()),
            'colorscale': palette['surface'],
            'showscale': True,
            'colorbar': {'title': {'text': mapped.colorbar or '', 'font': {'size': 10}},
                         'thickness': 10, 'len': 0.6, 'x': 1.01,
                         'tickfont': {'size': 9, 'color': palette['muted']},
                         'outlinewidth': 0},
        })
    else:
        marker['color'] = palette['accent']

    traces.append({
        'type': 'scattergl', 'mode': 'markers',
        'x': _to_list(x), 'y': _to_list(y), 'marker': marker,
        'hovertemplate': '%{x:.3g}, %{y:.3g}<extra></extra>',
        'showlegend': False,
    })

    layout = {
        'xaxis': {**_axis(palette, mapped.axes[0] if mapped.axes else '', x_range),
                  'showgrid': False, 'zeroline': False},
        'yaxis': {**_axis(palette, mapped.axes[1] if len(mapped.axes) > 1 else '', y_range),
                  'showgrid': False, 'zeroline': False,
                  'scaleanchor': 'x', 'scaleratio': 1},
    }
    return traces, layout


def _render_conformal(mapped: Mapped, palette: dict[str, Any]) -> tuple[list[dict], dict]:
    """Grid lines of the z-plane, drawn where f maps them."""
    traces = []
    for key, colour in (('rows', palette['accent']), ('cols', palette['accent2'])):
        lines = mapped.data[key]
        line_x, line_y = _polyline((np.real(row), np.imag(row)) for row in lines)
        traces.append({
            'type': 'scattergl', 'mode': 'lines',
            'x': _to_list(line_x), 'y': _to_list(line_y),
            'line': {'color': colour, 'width': 1.2},
            'hoverinfo': 'skip', 'showlegend': False,
        })

    all_x = np.concatenate([np.real(mapped.data[k]).ravel() for k in ('rows', 'cols')])
    all_y = np.concatenate([np.imag(mapped.data[k]).ravel() for k in ('rows', 'cols')])
    x_range, y_range = _match_aspect(_robust_range(all_x), _robust_range(all_y))
    layout = {
        'xaxis': _axis(palette, mapped.axes[0], x_range),
        'yaxis': {**_axis(palette, mapped.axes[1], y_range),
                  'scaleanchor': 'x', 'scaleratio': 1},
    }
    return traces, layout


def _render_polar(mapped: Mapped, palette: dict[str, Any]) -> tuple[list[dict], dict]:
    radius = mapped.data['r']
    traces = [{
        'type': 'scatterpolargl' if radius.size > WEBGL_THRESHOLD else 'scatterpolar',
        'mode': 'lines',
        'r': _to_list(radius), 'theta': _to_list(mapped.data['theta']),
        'line': {'color': palette['accent3'], 'width': 2.2},
        'hovertemplate': 'r %{r:.4g}, θ %{theta:.1f}°<extra></extra>',
        'showlegend': False,
    }]
    finite = radius[np.isfinite(radius)]
    upper = float(np.percentile(finite, 99.5)) * 1.08 if finite.size else 1.0
    layout = {
        'polar': {
            'bgcolor': palette['plot'],
            'radialaxis': {'range': [0, upper or 1.0], 'gridcolor': palette['grid'],
                           'linecolor': palette['line'], 'angle': 67.5,
                           'tickfont': {'size': 9, 'color': palette['muted']}},
            'angularaxis': {'direction': 'counterclockwise', 'gridcolor': palette['grid'],
                            'linecolor': palette['line'],
                            'tickfont': {'size': 9, 'color': palette['muted']}},
        },
        'margin': {'l': 30, 'r': 30, 't': 22, 'b': 22},
    }
    return traces, layout


def _render_scene(mapped: Mapped, palette: dict[str, Any]) -> tuple[list[dict], dict]:
    x, y, z = mapped.data['x'], mapped.data['y'], mapped.data['z']
    colour = mapped.data.get('color')
    arrays = _decimate(*(a for a in (x, y, z, colour) if a is not None))
    x, y, z = arrays[0], arrays[1], arrays[2]
    colour = arrays[3] if len(arrays) > 3 else None

    surface: dict[str, Any] = {
        'type': 'surface',
        'x': _to_list(x), 'y': _to_list(y), 'z': _to_list(z),
        'colorscale': CYCLIC_SCALE if mapped.colorbar == 'arg f' else palette['surface'],
        'colorbar': {'title': {'text': mapped.colorbar or '', 'font': {'size': 10}},
                     'thickness': 10, 'len': 0.6, 'x': 0.94,
                     'tickfont': {'size': 9, 'color': palette['muted']},
                     'outlinewidth': 0},
        'showscale': True,
        'contours': {'z': {'show': False}},
        'lighting': {'ambient': 0.72, 'diffuse': 0.82, 'specular': 0.12, 'roughness': 0.55},
        'hovertemplate': '%{x:.3g}, %{y:.3g}, %{z:.3g}<extra></extra>',
    }
    if colour is not None:
        surface['surfacecolor'] = _to_list(colour)

    traces: list[dict] = [surface]
    if mapped.overlay == 'sphere-grid':
        traces.append(_sphere_grid(palette))

    visible = bool(mapped.axes and any(mapped.axes))
    titles = mapped.axes if visible else ('', '', '')
    layout = {
        'scene': {
            'xaxis': _scene_axis(palette, titles[0], visible),
            'yaxis': _scene_axis(palette, titles[1], visible),
            'zaxis': _scene_axis(palette, titles[2], visible),
            'aspectmode': 'data' if mapped.aspect == 'force' else 'auto',
            'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.1}},
            # Drag to rotate. Stated explicitly because Plotly would otherwise
            # inherit layout.dragmode ('pan') as this scene's default.
            'dragmode': 'turntable',
            'bgcolor': palette['plot'],
        },
        'margin': {'l': 4, 'r': 4, 't': 4, 'b': 4},
    }
    return traces, layout


def _render_image(mapped: Mapped, palette: dict[str, Any]) -> tuple[list[dict], dict]:
    rgb = mapped.data['rgb']
    x_low, x_high = (float(v) for v in mapped.data['x'])
    y_low, y_high = (float(v) for v in mapped.data['y'])
    height, width = rgb.shape[:2]

    traces = [{
        'type': 'image',
        'source': _png_data_uri(rgb),
        'x0': x_low, 'dx': (x_high - x_low) / max(width - 1, 1),
        'y0': y_low, 'dy': (y_high - y_low) / max(height - 1, 1),
        'hovertemplate': 'z = %{x:.3g} %{y:+.3g}i<extra></extra>',
    }]
    layout = {
        'xaxis': {**_axis(palette, mapped.axes[0], [x_low, x_high]),
                  'showgrid': False, 'zeroline': False, 'constrain': 'domain'},
        'yaxis': {**_axis(palette, mapped.axes[1], [y_low, y_high]),
                  'showgrid': False, 'zeroline': False,
                  'autorange': True, 'scaleanchor': 'x', 'scaleratio': 1},
        'margin': {'l': 50, 'r': 18, 't': 14, 'b': 40},
    }
    return traces, layout


_RENDERERS = {
    'polar': _render_polar,
    'scene': _render_scene,
    'image': _render_image,
}


def build(sample: Sample, transform: Transform, theme_name: str = 'light') -> dict[str, Any]:
    """Apply `transform` to `sample` and return a Plotly figure dictionary."""
    palette = theme(theme_name)

    if sample.domain == 'field':
        mapped = transform.apply(sample.coords['x'], sample.coords['y'], sample.extras['w'])
    elif sample.domain == 'surface':
        mapped = transform.apply(sample.coords['x'], sample.coords['y'], sample.coords['z'])
    else:
        mapped = transform.apply(sample.coords['x'], sample.coords['y'])

    if mapped.aspect == 'source':
        mapped.aspect = 'force' if sample.isotropic else 'free'

    if mapped.render == 'xy':
        if mapped.overlay == 'conformal':
            traces, extra = _render_conformal(mapped, palette)
        elif sample.domain == 'curve':
            traces, extra = _render_xy_curve(mapped, palette)
        else:
            traces, extra = _render_xy_scatter(mapped, palette)
    else:
        traces, extra = _RENDERERS[mapped.render](mapped, palette)

    layout = _base_layout(palette)
    layout.update(extra)
    return {
        'key': transform.key,
        'label': transform.label,
        'note': mapped.note or transform.summary,
        'render': mapped.render,
        'equations': list(transform.equations),
        'data': traces,
        'layout': layout,
    }
