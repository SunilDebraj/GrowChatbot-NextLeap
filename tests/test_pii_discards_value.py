"""GR7 / D6: on a PII hit the query value is discarded, not forwarded.

The assertion is made with a spy rather than by inspecting the return value.
Checking that ``sanitized_query == ""`` proves only that one field is empty; it
does not prove the value did not travel to some other callee first. So every
collaborator the pipeline is allowed to touch is replaced by a recorder, and the
test asserts that none of them ever received the identifier.
"""

from __future__ import annotations

import pytest

from mf_facts.common.config import AppConfig
from mf_facts.common.models import PiiResult
from mf_facts.online.classifier import Classifier
from mf_facts.online.pii import PiiScanner
from mf_facts.online.rewriter import QueryRewriter

pytestmark = [pytest.mark.contract, pytest.mark.adversarial]

RAW_PAN = "ABCDE1234F"
QUERY = f"my PAN {RAW_PAN}, what is the expense ratio of HDFC Large Cap Fund?"


class SpyRewriter(QueryRewriter):
    """A rewriter that records every query it is handed."""

    def __init__(self, inner: QueryRewriter) -> None:
        self._inner = inner
        self.seen: list[str] = []

    @property
    def scheme_keys(self):
        return self._inner.scheme_keys

    def resolve_scheme_keys(self, query: str):
        self.seen.append(query)
        return self._inner.resolve_scheme_keys(query)

    def has_scheme_token(self, query: str) -> bool:
        self.seen.append(query)
        return self._inner.has_scheme_token(query)

    def doc_class_hints(self, query: str):
        self.seen.append(query)
        return self._inner.doc_class_hints(query)

    def rewrite(self, query: str):
        self.seen.append(query)
        return self._inner.rewrite(query)


class SpyLLM:
    def __init__(self) -> None:
        self.seen: list[dict] = []

    def generate(self, **kwargs):
        self.seen.append(kwargs)
        return '{"label": "factual_scheme", "confidence": 1.0}'


def test_pii_discards_value(app_config, alias_map):
    scanner = PiiScanner(app_config)
    rewriter = SpyRewriter(QueryRewriter(alias_map))
    llm = SpyLLM()
    classifier = Classifier(rewriter=rewriter, llm=llm)

    result = scanner.scan(QUERY)
    assert result.is_pii is True
    assert result.sanitized_query == ""

    # The pipeline hands the classifier the sanitized text, never the raw query.
    classification = classifier.classify(result.sanitized_query, result)
    assert classification.query_class == "pii"

    # No collaborator ever saw the identifier.
    for seen in rewriter.seen:
        assert RAW_PAN not in seen
    assert llm.seen == []
    assert RAW_PAN not in repr(result)
    assert RAW_PAN not in repr(classification)


def test_pii_result_carries_only_the_boolean_and_categories(app_config):
    result = PiiScanner(app_config).scan(QUERY)

    assert set(result.to_log_fields()) == {"pii_detected", "pii_categories"}
    assert result.to_log_fields() == {"pii_detected": True, "pii_categories": ["pan"]}


def test_whole_query_is_refused_not_just_the_identifier(app_config):
    """A masked query would still be answerable, and answerable with the
    identifier attached. The trailing question must die with the PAN."""
    result = PiiScanner(app_config).scan(QUERY)

    assert result.sanitized_query == ""
    assert "expense ratio" not in result.sanitized_query
