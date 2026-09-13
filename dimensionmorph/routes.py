"""HTTP layer: one catalogue endpoint and one plotting endpoint."""

from __future__ import annotations

import re
import time
from typing import Any, Mapping, cast

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory

from . import figures, sources, transforms
from .expr import ExpressionError, function_catalog

bp = Blueprint('dimensionmorph', __name__)

MAX_PARAMETERS = 24
MAX_VIEWS = 6
_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,19}$')
_RESERVED = {'x', 'y', 'z', 't', 'u', 'v', 'w'}


class RequestError(ValueError):
    """A malformed request, reported to the client as HTTP 400."""


def _clean_parameters(raw: Any) -> dict[str, float]:
    if raw in (None, ''):
        return {}
    if not isinstance(raw, Mapping):
        raise RequestError('parameters must be an object')
    if len(raw) > MAX_PARAMETERS:
        raise RequestError(f'at most {MAX_PARAMETERS} parameters are supported')

    cleaned: dict[str, float] = {}
    for name, value in raw.items():
        name = str(name).strip()
        if not _NAME.match(name):
            raise RequestError(f'"{name}" is not a valid parameter name')
        if name in _RESERVED:
            raise RequestError(f'"{name}" is reserved for coordinates')
        try:
            cleaned[name] = float(value)
        except (TypeError, ValueError):
            raise RequestError(f'parameter "{name}" is not a number') from None
    return cleaned


def _clean_views(raw: Any, domain: str) -> list[str]:
    available = {t.key for t in transforms.for_domain(domain)}
    requested = [str(v) for v in raw] if isinstance(raw, (list, tuple)) else []
    views = [key for key in requested if key in available][:MAX_VIEWS]
    return views or list(transforms.DEFAULT_VIEWS[domain])


@bp.get('/')
def index():
    return render_template('index.html')


@bp.get('/icon.png')
def icon():
    return send_from_directory(current_app.root_path, 'icon.png', mimetype='image/png')


@bp.get('/api/catalog')
def catalog():
    """Everything the client needs to build its form: sources, views, functions."""
    response = jsonify({
        'sources': sources.catalog(),
        'transforms': transforms.catalog(),
        'functions': function_catalog(),
        'themes': list(figures.THEMES),
    })
    response.headers['Cache-Control'] = 'public, max-age=300'
    return response


@bp.post('/api/plot')
def plot():
    started = time.perf_counter()
    payload = request.get_json(silent=True)
    if not isinstance(payload, Mapping):
        return jsonify({'ok': False, 'error': 'expected a JSON object'}), 400

    try:
        source = sources.get(str(payload.get('source', 'explicit2d')))
        parameters = _clean_parameters(payload.get('parameters'))
        expressions = payload.get('expressions')
        if not isinstance(expressions, Mapping):
            raise RequestError('expressions must be an object')
        ranges = cast(
            Mapping[str, Mapping[str, Any]],
            payload.get('ranges') if isinstance(payload.get('ranges'), Mapping) else {},
        )

        sampled = sources.sample(
            source, expressions, ranges, payload.get('samples', source.samples), parameters,
        )
        view_keys = _clean_views(payload.get('views'), source.domain)
        theme = str(payload.get('theme', 'light'))
        built = [
            figures.build(sampled, transforms.get(source.domain, key), theme)
            for key in view_keys
        ]
    except KeyError as exc:
        # str(KeyError) quotes its argument; the message is the argument itself.
        return jsonify({'ok': False, 'error': str(exc.args[0])}), 400
    except (RequestError, ExpressionError) as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400
    except (ValueError, TypeError, ArithmeticError) as exc:
        return jsonify({'ok': False, 'error': f'could not plot: {exc}'}), 400

    return jsonify({
        'ok': True,
        'domain': source.domain,
        'caption': sampled.caption,
        'figures': built,
        'elapsedMs': round((time.perf_counter() - started) * 1000, 1),
    })
