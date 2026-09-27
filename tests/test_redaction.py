"""Log-time redaction (P0-T11)."""
from __future__ import annotations

from mf_facts.common.redaction import MASK, contains_pii, redact
import pytest

pytestmark = pytest.mark.contract




def test_pan_is_masked():
    clean, categories = redact("my PAN is ABCDE1234F")
    assert "ABCDE1234F" not in clean
    assert MASK in clean
    assert categories == ["pan"]


def test_email_is_masked():
    clean, categories = redact("write to ravi@example.com please")
    assert "ravi@example.com" not in clean
    assert categories == ["email"]


def test_multiple_values_are_all_masked():
    clean, categories = redact("pan ABCDE1234F, otp 448291, mail a@b.com")
    assert "ABCDE1234F" not in clean
    assert "448291" not in clean
    assert "a@b.com" not in clean
    assert set(categories) == {"pan", "otp", "email"}


def test_clean_text_is_untouched():
    text = "expense ratio 1.03% and minimum SIP Rs 500"
    clean, categories = redact(text)
    assert clean == text
    assert categories == []


def test_contains_pii():
    assert contains_pii("pan ABCDE1234F")
    assert not contains_pii("what is the exit load")


def test_empty_input():
    assert redact("") == ("", [])
