"""No PII at rest (architecture.md section 14.1, PRD Non-Goal).

Scans the surfaces that could hold a *user's* identifier for the PII patterns the
query-time scanner uses, so that a value which should have been discarded cannot
turn up in something written to disk.

Scope, and why it is not the whole tree:

* Scanned: ``src/``, ``config/``, ``assets/``, ``artifacts/``, the vector store,
  every ``*.log``, plus ``tests/`` and ``eval/`` under a synthetic-value
  allowlist. This is the union of places this codebase can write to at query
  time, plus the build artifacts.
* Exempt: the three specification documents at the repository root. They *quote*
  the detection patterns on purpose - a sample PAN, a sample address - and a
  spec that documents a pattern is not leaking one. Exempting them is stated
  here rather than done silently.
* Exempt: ``raw_cache/`` and ``cache/``. ``raw_cache`` holds public scheme pages
  fetched by the offline pipeline, which carry no user data by construction (P1
  verified zero PII in the indexed corpus). ``cache/embeddings`` holds
  MiniLM vectors as decimal literals, where a 12-digit run is a coordinate and
  not an account number; scanning it would fail at random.
* Long hex digests are removed before scanning. A ``content_hash`` is 64 hex
  characters and a 12-digit run inside one is a roughly 1-in-270 coincidence per
  position; without this the test would fail intermittently and be ignored. This
  is the one place the scanner is weakened, so it is documented rather than
  buried.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mf_facts.common.pii_patterns import find_pii

pytestmark = [pytest.mark.contract, pytest.mark.adversarial]

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every identifier-shaped string permitted to exist on disk, and only inside
# tests/ and eval/. The list is deliberately explicit rather than a category
# check: adding a new PII-looking literal to a fixture then fails this test and
# forces someone to confirm the value is synthetic. That friction is the control.
_SYNTHETIC = {
    # P3 routing fixtures.
    "ABCDE1234F", "XXXXX1234F", "XXXX1234X", "2345 6789 0123", "234567890123",
    "1234567890", "1234-5678-9012", "123456789012", "9876543210",
    "482913", "otp is 482913", "priya.sharma@example.com",
    # P1 detection/redaction fixtures that predate this test.
    "XXXXX1234X", "+91 98765 43210", "otp is 448291", "otp 448291",
    "verification code: 123456", "ravi@example.com", "a@b.com", "0123456789",
}

# Directories holding synthetic PII by design.
_ALLOWED_ROOTS = ("tests", "eval")

# Runtime surfaces. Must exist for the scan to be non-vacuous.
_SCAN_ROOTS = ("src", "config", "assets", "artifacts", "tests", "eval")

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".venv", "venv", "env", "node_modules", ".idea", ".vscode",
}

_LONG_HEX = re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE)
_TEXT_SUFFIXES = {
    ".py", ".yaml", ".yml", ".json", ".jsonl", ".md", ".txt", ".csv", ".log",
    ".toml", ".cfg", ".ini", ".env", "",
}


def _iter_files():
    for name in _SCAN_ROOTS:
        root = REPO_ROOT / name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not any(p in _SKIP_DIRS for p in path.parts):
                if path.suffix.lower() in _TEXT_SUFFIXES:
                    yield path
    # Any log anywhere in the tree, including outside the scan roots.
    for path in REPO_ROOT.rglob("*.log"):
        if path.is_file() and not any(p in _SKIP_DIRS for p in path.parts):
            yield path


def _scan(text: str) -> list[tuple[str, str]]:
    scrubbed = _LONG_HEX.sub("<hash>", text)
    return [(hit.category, hit.text) for hit in find_pii(scrubbed)]


def test_scan_surfaces_exist():
    """Guards the scanner: if these paths vanished, the scan would pass vacuously."""
    missing = [name for name in _SCAN_ROOTS if not (REPO_ROOT / name).is_dir()]
    assert not missing, f"scan surfaces missing: {missing}"


def test_no_pii_at_rest():
    offenders: list[str] = []
    unexpected: list[str] = []
    scanned = 0

    for path in _iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        rel = path.relative_to(REPO_ROOT)
        hits = _scan(text)
        if not hits:
            continue
        if rel.parts and rel.parts[0] in _ALLOWED_ROOTS:
            for _, value in hits:
                if value not in _SYNTHETIC:
                    unexpected.append(f"{rel}: unrecognised PII {value!r}")
            continue
        offenders.append(f"{rel}: {hits}")

    assert scanned > 20, f"only {scanned} files scanned; the sweep is not running"
    assert not offenders, "PII found in a runtime surface:\n" + "\n".join(offenders)
    assert not unexpected, "unrecognised PII in a fixtures directory:\n" + "\n".join(
        unexpected
    )


def test_config_files_carry_no_identifier():
    for rel in (
        "config/config.yaml",
        "config/sources.yaml",
        "config/aliases.yaml",
        "config/educational_links.yaml",
    ):
        path = REPO_ROOT / rel
        assert path.is_file(), f"{rel} is missing; the PII scan cannot cover it"
        assert not _scan(path.read_text(encoding="utf-8")), rel


def test_online_modules_carry_no_identifier():
    for path in (REPO_ROOT / "src" / "mf_facts" / "online").rglob("*.py"):
        assert not _scan(path.read_text(encoding="utf-8")), path.name


def test_declared_synthetic_values_are_actually_covered():
    """If a new fixture value is added to a test but not to _SYNTHETIC, the
    fixtures-directory check above will fail. This asserts the allowlist is
    still meaningful rather than trivially empty."""
    assert len(_SYNTHETIC) >= 10
