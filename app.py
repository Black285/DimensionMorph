"""Entry point.  Run `python app.py`, or serve `dimensionmorph:create_app()`."""

from __future__ import annotations

import os

from dimensionmorph import create_app

app = create_app()

if __name__ == '__main__':
    app.run(
        host=os.environ.get('DM_HOST', '127.0.0.1'),
        port=int(os.environ.get('DM_PORT', '5000')),
        debug=os.environ.get('DM_DEBUG', '1') == '1',
    )
