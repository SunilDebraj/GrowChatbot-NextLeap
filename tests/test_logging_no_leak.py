"""No PII reaches a log handler (P0-T11, architecture.md section 14).

Two independent protections are tested: the deny-by-default field allowlist, and
the formatter-level redactor that covers arbitrary log calls.
"""
from __future__ import annotations

import io
import logging

from mf_facts.common.logging import RedactingFormatter, log_event, setup_logging
import pytest

pytestmark = pytest.mark.contract






def _capture(logger_name: str = "mf_facts.test"):
    stream = io.StringIO()
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, stream


def test_forbidden_field_is_dropped_by_the_allowlist():
    logger, stream = _capture()
    log_event(logger, "query.handled", query="my PAN is ABCDE1234F", route="factual")
    output = stream.getvalue()
    assert "ABCDE1234F" not in output
    assert "factual" in output


def test_unknown_field_is_dropped():
    logger, stream = _capture()
    log_event(logger, "e", some_random_field="ABCDE1234F")
    assert "ABCDE1234F" not in stream.getvalue()


def test_formatter_redacts_a_raw_log_call():
    """A careless logger.warning must still not write a PAN to disk."""
    logger, stream = _capture()
    logger.warning("user typed %s", "my PAN is ABCDE1234F")
    output = stream.getvalue()
    assert "ABCDE1234F" not in output
    assert "REDACTED" in output


def test_formatter_redacts_tracebacks():
    logger, stream = _capture()
    try:
        raise ValueError("failed for pan ABCDE1234F")
    except ValueError:
        logger.exception("boom")
    assert "ABCDE1234F" not in stream.getvalue()


def test_allowed_fields_survive():
    logger, stream = _capture()
    log_event(
        logger,
        "query.handled",
        route="performance",
        rule_id="ref:performance:cagr",
        latency_ms=123,
        pii_detected=False,
    )
    output = stream.getvalue()
    assert "performance" in output
    assert "ref:performance:cagr" in output
    assert "123" in output


def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    root = logging.getLogger()
    first = len(root.handlers)
    setup_logging("INFO")
    assert len(root.handlers) == first == 1
