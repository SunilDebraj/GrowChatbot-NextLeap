"""Configuration loading is strict (P0-T11)."""

from __future__ import annotations

import pytest

from mf_facts.common.config import LEGAL_CHUNKING_STRATEGIES, load_config
from mf_facts.common.errors import ConfigError

pytestmark = pytest.mark.contract






def test_config_loads(app_config):
    assert app_config.embedding.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert app_config.embedding.embedding_dim == 384
    assert app_config.corpus.collection == "mf_facts_v1"
    assert app_config.chunking.strategy in LEGAL_CHUNKING_STRATEGIES
    assert len(app_config.ui.examples) == 3


def test_token_cap_is_below_the_model_limit(app_config):
    """all-MiniLM-L6-v2 truncates at 256 tokens, so the cap must be under it."""
    assert app_config.chunking.max_chunk_tokens <= 256
    assert app_config.chunking.target_tokens <= app_config.chunking.max_chunk_tokens


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "corpus:\n  name: x\n  nonsense_key: 1\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(path)


def test_illegal_strategy_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("chunking:\n  strategy: vibes\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not legal"):
        load_config(path)


def test_cap_above_model_limit_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("chunking:\n  max_chunk_tokens: 400\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="exceeds the all-MiniLM-L6-v2"):
        load_config(path)


def test_target_above_cap_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "chunking:\n  max_chunk_tokens: 100\n  target_tokens: 200\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="must not exceed"):
        load_config(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_performance_exclusions_are_configured(app_config):
    """D5 / GR6: the returns regions must be listed for the parser to drop them."""
    joined = " ".join(app_config.safety.performance_exclude_patterns).lower()
    assert "returns and rankings" in joined
    assert "return calculator" in joined
    assert "compare similar funds" in joined


def test_log_raw_queries_defaults_off(app_config):
    assert app_config.safety.log_raw_queries is False
    assert app_config.safety.pii_scan_enabled is True
