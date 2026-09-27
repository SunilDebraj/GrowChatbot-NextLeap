"""``python -m api`` - serve the prototype.

``uvicorn.run`` is imported inside ``api.app.main`` rather than at module import,
so that importing the app in a test does not require an ASGI server to be
installed.
"""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
