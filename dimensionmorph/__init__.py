"""DimensionMorph -- visualise a function under a change of coordinates."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

__version__ = '2.0.0'

_ROOT = Path(__file__).resolve().parent.parent


def create_app(**overrides: object) -> Flask:
    """Application factory."""
    app = Flask(
        __name__,
        root_path=str(_ROOT),
        static_folder=str(_ROOT / 'static'),
        template_folder=str(_ROOT / 'templates'),
        static_url_path='/static',
    )
    app.config.update(
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=256 * 1024,
        SEND_FILE_MAX_AGE_DEFAULT=int(os.environ.get('DM_STATIC_MAX_AGE', 0)),
        VERSION=__version__,
    )
    app.config.update(overrides)

    from .routes import bp

    app.register_blueprint(bp)
    return app
