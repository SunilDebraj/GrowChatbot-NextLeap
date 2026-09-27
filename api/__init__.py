"""P5 - the HTTP surface and the demo page.

Three endpoints (architecture 5.19) and one page. See ``api.app`` for why this is
hand-rolled ASGI rather than a framework.
"""

from .app import FactsApp, RateLimiter, build_app, main

__all__ = ["FactsApp", "RateLimiter", "build_app", "main"]
