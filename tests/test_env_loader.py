"""Tests for the .env reader.

GR14: nothing here reads the repo's real .env, and nothing touches the network.
A test that loaded the developer's actual key would both be non-reproducible and
be one stray assertion away from printing a secret into CI logs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mf_facts.common.env import is_loaded, load_env, parse_env_text, reset


@pytest.fixture(autouse=True)
def _clean_env(tmp_path):
    """Isolate os.environ so no test can see or disturb a real key."""
    saved = dict(os.environ)
    reset()
    os.environ.pop("MF_TEST_ENV_VAR", None)
    yield
    os.environ.clear()
    os.environ.update(saved)
    reset()


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parses_plain_assignments():
    assert parse_env_text("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_strips_quotes_and_export_prefix():
    text = 'export A="one two"\nB=\'three\'\n'
    assert parse_env_text(text) == {"A": "one two", "B": "three"}


def test_ignores_comments_blank_lines_and_junk():
    text = "# a comment\n\nA=1\nnot an assignment\n   \nB=2\n"
    assert parse_env_text(text) == {"A": "1", "B": "2"}


def test_keeps_equals_signs_inside_the_value():
    """A key is a provider token; a value can legitimately contain '='."""
    assert parse_env_text("A=abc=def==\n") == {"A": "abc=def=="}


def test_keeps_a_hash_that_is_part_of_a_value():
    """A '#' only starts a comment at the start of a line, not mid-value.

    Naive parsers strip everything after a '#', which silently mangles values
    containing one.
    """
    assert parse_env_text("A=sk-123#456\n") == {"A": "sk-123#456"}


def test_handles_empty_value_and_bom():
    assert parse_env_text("﻿A=\n") == {"A": ""}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}
    assert is_loaded() is True


def test_loads_into_environ(tmp_path):
    write(tmp_path, "MF_TEST_ENV_VAR=secret-value\n")
    applied = load_env(tmp_path / ".env")

    assert applied == {"MF_TEST_ENV_VAR": "secret-value"}
    assert os.environ["MF_TEST_ENV_VAR"] == "secret-value"


def test_real_environment_wins_over_the_file(tmp_path):
    """The whole point of a .env: a shell or CI secret must not need a file edit."""
    os.environ["MF_TEST_ENV_VAR"] = "from-shell"
    write(tmp_path, "MF_TEST_ENV_VAR=from-file\n")

    load_env(tmp_path / ".env")

    assert os.environ["MF_TEST_ENV_VAR"] == "from-shell"


def test_override_replaces_an_existing_value(tmp_path):
    os.environ["MF_TEST_ENV_VAR"] = "from-shell"
    write(tmp_path, "MF_TEST_ENV_VAR=from-file\n")

    load_env(tmp_path / ".env", override=True)

    assert os.environ["MF_TEST_ENV_VAR"] == "from-file"


def test_unreadable_file_is_not_fatal(tmp_path):
    """A directory named .env is a plausible mistake; it must not crash the boot.

    The provider then fails closed with a clear "needs $VAR" message, which is
    far more useful than a traceback from the loader.
    """
    (tmp_path / ".env").mkdir()
    assert load_env(tmp_path / ".env") == {}


# --------------------------------------------------------------------------
# the real file
# --------------------------------------------------------------------------


def test_repo_dot_env_is_gitignored():
    """A committed .env means a committed key, and the key is already in a chat log."""
    repo = Path(__file__).resolve().parents[1]
    ignore = (repo / ".gitignore").read_text(encoding="utf-8")

    assert ".env" in ignore


def test_no_key_is_hardcoded_in_the_source():
    """A provider key literal in tracked source is a leaked key.

    Deliberately a substring test rather than an exact-value test: it fails for
    *any* key pasted into source, not just the one that exists today. The needle
    is assembled at runtime because this file is itself scanned, and writing the
    literal here would make it match itself forever.
    """
    repo = Path(__file__).resolve().parents[1]
    needle = "gsk" + "_"
    tracked = [
        p
        for p in repo.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".yaml", ".yml", ".toml", ".md", ".js", ".css", ".txt"}
        and ".venv" not in p.parts
        and "__pycache__" not in p.parts
        and p.name != ".env"
    ]

    offenders = [
        str(p.relative_to(repo))
        for p in tracked
        if needle in p.read_text(encoding="utf-8", errors="ignore")
    ]

    assert not offenders, f"API key literal found in: {offenders}"
