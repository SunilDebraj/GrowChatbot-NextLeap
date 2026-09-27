"""Chunking strategy implementations, selected by config.

The strategy is resolved from ``chunking.strategy`` in config.yaml and sits behind
the ``ChunkingStrategy`` protocol, so the P2 decision is reversible without
touching retrieval or generation.
"""

from __future__ import annotations

from ...common.config import ChunkingConfig
from ...common.errors import ConfigError
from ..chunker import BaseStrategy, ChunkingStrategy
from .atomic_fact import AtomicFactStrategy
from .fixed_window import FixedWindowStrategy
from .heading_aware import HeadingAwareStrategy

STRATEGIES: dict[str, type[BaseStrategy]] = {
    HeadingAwareStrategy.name: HeadingAwareStrategy,
    FixedWindowStrategy.name: FixedWindowStrategy,
    AtomicFactStrategy.name: AtomicFactStrategy,
}

__all__ = [
    "AtomicFactStrategy",
    "BaseStrategy",
    "ChunkingStrategy",
    "FixedWindowStrategy",
    "HeadingAwareStrategy",
    "STRATEGIES",
    "build_strategy",
]


def build_strategy(
    config: ChunkingConfig, count_tokens=None, name: str | None = None
) -> BaseStrategy:
    """Instantiate the configured strategy, or one named explicitly.

    ``name`` overrides ``config.strategy`` so the P2 harness can build the same
    corpus under all three strategies without rewriting the config file.
    """
    selected = name or config.strategy
    try:
        strategy_class = STRATEGIES[selected]
    except KeyError as exc:
        raise ConfigError(
            f"unknown chunking strategy {selected!r}; available: {sorted(STRATEGIES)}"
        ) from exc
    return strategy_class(config, count_tokens=count_tokens)
