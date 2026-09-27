"""A minimal ``.env`` reader, so a provider key can stay out of the code.

There is no ``python-dotenv`` in this project's dependencies and adding one for
twelve lines of parsing would be a poor trade, so this reads the file directly.
The rules it follows are the ones that actually matter:

* **A real environment variable always wins.** A value already in ``os.environ``
  is never overwritten, so CI, Docker and a developer's shell override the file
  without anyone editing it.
* **Values are never logged, echoed or returned in error messages.** The only
  thing this module will tell you about a key is whether it is *set*.
* **The file is optional.** No ``.env`` is a normal state for this project -
  every test runs without one - so a missing file is not an error.

``implementation.md`` 2.3 requires secrets to come from the environment, never
from a tracked file. ``.gitignore`` lists ``.env`` for exactly this reason.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set once a load has been attempted, so a later call is a cheap no-op and a
#: value injected by a test after the first load is not silently overwritten.
_loaded = False


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines into a dict.

    Handles the forms that appear in real ``.env`` files and ignores the rest:
    blank lines, ``#`` comments, an optional ``export`` prefix, and single or
    double quoted values. A line with no ``=`` is skipped rather than raising,
    because a stray comment should not stop the service from starting.
    """
    values: dict[str, str] = {}
    for raw in text.lstrip("﻿").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into ``os.environ``. Returns the values that were applied.

    ``override=False`` (the default) is the behaviour to keep: an already-set
    environment variable is left alone.
    """
    global _loaded

    target = Path(path) if path else _default_env_path()
    if not target.is_file():
        _loaded = True
        return {}

    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError:
        # An unreadable .env must not be fatal. The provider will fail closed with
        # a clear "needs $VAR" message, which is more useful than a boot crash.
        _loaded = True
        return {}

    applied: dict[str, str] = {}
    for key, value in parse_env_text(text).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value

    _loaded = True
    return applied


def is_loaded() -> bool:
    return _loaded


def reset() -> None:
    """Forget that a load happened. For tests."""
    global _loaded
    _loaded = False


def _default_env_path() -> Path:
    # env.py lives in src/mf_facts/common/, so the repo root is three directories
    # up from its own directory: common/ -> mf_facts/ -> src/ -> <repo root>.
    return Path(__file__).resolve().parents[3] / ".env"
