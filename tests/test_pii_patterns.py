"""PII pattern detection (P0-T11)."""

from __future__ import annotations

import pytest

from mf_facts.common.pii_patterns import categories_in, find_pii

pytestmark = pytest.mark.contract




@pytest.mark.parametrize(
    "text,expected",
    [
        ("my PAN is ABCDE1234F", "pan"),
        ("PAN ABCDE1234F please help", "pan"),
        ("XXXXX1234X is my pan", "pan"),
    ],
)
def test_pan_detected(text, expected):
    assert expected in categories_in(text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("aadhaar 2345 6789 0123", "aadhaar"),
        ("aadhaar 234567890123", "aadhaar"),
    ],
)
def test_aadhaar_detected(text, expected):
    categories = categories_in(text)
    assert expected in categories


def test_aadhaar_wins_over_account_for_twelve_digits():
    hits = find_pii("my aadhaar is 234567890123 ok")
    assert [hit.category for hit in hits] == ["aadhaar"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("call me on +91 98765 43210", "phone"),
        ("9876543210 is my number", "phone"),
        ("my otp is 448291", "otp"),
        ("verification code: 123456", "otp"),
        ("email me at ravi@example.com", "email"),
        ("account 1234567890 please", "account"),
    ],
)
def test_other_categories(text, expected):
    assert expected in categories_in(text)


@pytest.mark.parametrize(
    "text",
    [
        "what is the expense ratio",
        "minimum SIP of 500 rupees",
        "the lock-in period is 3 years",
        "exit load of 1% if redeemed within 1 year",
        "NIFTY 100 Total Return Index is the benchmark",
        "is HDFC Large Cap a good fund",
    ],
)
def test_fund_questions_are_not_pii(text):
    """Numeric fund facts must not trip the scanner.

    This is the false-positive risk that would make the assistant unusable: a
    minimum SIP amount, a lock-in duration and an expense ratio are all digits.
    """
    assert find_pii(text) == [], categories_in(text)


def test_bare_four_to_six_digit_number_is_not_an_otp():
    assert "otp" not in categories_in("the minimum is 500 per month")


def test_pii_embedded_mid_question_is_found():
    categories = categories_in("my PAN ABCDE1234F, what is the expense ratio?")
    assert "pan" in categories


def test_overlapping_patterns_yield_one_hit():
    hits = find_pii("aadhaar 234567890123")
    assert len(hits) == 1
