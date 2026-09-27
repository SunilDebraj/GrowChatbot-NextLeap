"""Eval package: the chunking gate (architecture.md 5.10) and its report.

Deliberately free of eager submodule imports. ``python -m mf_facts.eval.harness``
is the documented entry point, and if this package imported ``harness`` at import
time the module would already be in sys.modules when runpy executed it, which
emits a RuntimeWarning and can produce two module objects with two sets of
globals. Import from ``mf_facts.eval.harness`` directly instead.
"""

from __future__ import annotations

__all__ = ["harness", "matching", "report"]
