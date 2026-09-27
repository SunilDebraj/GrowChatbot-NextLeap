"""Node [14] - alias normalizer (architecture.md section 5.14, PRD FR2 step 2).

Turns what a user typed into canonical ``scheme_key`` values plus soft
``doc_class`` hints. It introduces no facts: every output is either text the user
already wrote or a key read from ``config/aliases.yaml``.

The hints are deliberately a *ranking boost* and not a metadata filter. A hard
filter would silently produce "I don't have that in my sources" for any question
whose hint was wrong, and a wrong hint is a normal outcome - "exit load" is
genuinely a fees fact, but a user asking about the lock-in period in the same
breath pulls in faq/sid, and the corpus is not obliged to contain the answer for
the hint to be wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..common.errors import ConfigError
from ..common.models import DOC_CLASSES, RewrittenQuery

_WHITESPACE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").lower()).strip()


def _word_regex(phrase: str) -> re.Pattern[str]:
    """Match a cue as a whole phrase, not as a substring.

    Substring matching is not safe for cues this short: "ter" (Total Expense
    Ratio) occurs inside "riskometer", and "baf" or "elss" would occur inside any
    longer word a user happens to type. Anchoring on word boundaries is what
    keeps a soft hint from firing on a word that merely contains it.
    """
    return re.compile(rf"\b{re.escape(phrase)}\b")


@dataclass(frozen=True, slots=True)
class _Cue:
    doc_classes: tuple[str, ...]
    cues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AliasMap:
    """The parsed contents of ``config/aliases.yaml``."""

    scheme_keys: tuple[str, ...]
    names: dict[str, tuple[str, ...]]
    multi_scheme: tuple[str, ...]
    cues: tuple[_Cue, ...]

    @classmethod
    def load(cls, path: str | Path) -> AliasMap:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"alias map not found: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{config_path}: expected a mapping at the root")

        schemes_raw = raw.get("schemes")
        if not isinstance(schemes_raw, dict) or not schemes_raw:
            raise ConfigError(f"{config_path}: 'schemes' must be a non-empty mapping")

        names: dict[str, tuple[str, ...]] = {}
        for key, entry in schemes_raw.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("names"), list):
                raise ConfigError(f"{config_path}: schemes.{key}.names must be a list")
            aliases = tuple(_collapse(n) for n in entry["names"] if _collapse(str(n)))
            if not aliases:
                raise ConfigError(f"{config_path}: schemes.{key}.names is empty")
            names[str(key)] = aliases

        unknown_sections = sorted(set(raw) - {"schemes", "multi_scheme", "doc_class_cues"})
        if unknown_sections:
            raise ConfigError(f"{config_path}: unknown section(s) {unknown_sections}")

        multi = tuple(_collapse(m) for m in (raw.get("multi_scheme") or ()) if _collapse(str(m)))

        cues: list[_Cue] = []
        for entry in raw.get("doc_class_cues") or ():
            if not isinstance(entry, dict):
                raise ConfigError(f"{config_path}: each doc_class_cues entry must be a mapping")
            doc_classes = tuple(str(d) for d in entry.get("doc_classes") or ())
            bad = [d for d in doc_classes if d not in DOC_CLASSES]
            if not doc_classes or bad:
                raise ConfigError(
                    f"{config_path}: doc_class_cues has unknown doc_class(es) {bad}"
                )
            cue_words = tuple(_collapse(c) for c in entry.get("cues") or () if _collapse(str(c)))
            if not cue_words:
                raise ConfigError(f"{config_path}: doc_class_cues entry has no cues")
            cues.append(_Cue(doc_classes=doc_classes, cues=cue_words))

        return cls(
            scheme_keys=tuple(names),
            names=names,
            multi_scheme=multi,
            cues=tuple(cues),
        )


class QueryRewriter:
    """Resolves surface forms to canonical keys. No network, no model."""

    def __init__(self, alias_map: AliasMap) -> None:
        self._map = alias_map
        self._aliases = {
            key: tuple(_word_regex(a) for a in aliases)
            for key, aliases in alias_map.names.items()
        }
        self._multi = tuple(_word_regex(m) for m in alias_map.multi_scheme)
        self._cues = tuple(
            (cue.doc_classes, tuple(_word_regex(c) for c in cue.cues))
            for cue in alias_map.cues
        )

    @property
    def scheme_keys(self) -> tuple[str, ...]:
        return self._map.scheme_keys

    def resolve_scheme_keys(self, query: str) -> tuple[str, ...]:
        """Every scheme mentioned, in registry order. Never a single hard pick."""
        text = _collapse(query)
        if not text:
            return ()
        hits = [
            key
            for key, patterns in self._aliases.items()
            if any(p.search(text) for p in patterns)
        ]
        if not hits and any(p.search(text) for p in self._multi):
            return self._map.scheme_keys
        # Registry order, not match order, so the key list is stable across queries.
        return tuple(k for k in self._map.scheme_keys if k in hits)

    def doc_class_hints(self, query: str) -> tuple[str, ...]:
        """Union of every matching cue's doc classes, in cue order."""
        text = _collapse(query)
        hints: list[str] = []
        for doc_classes, patterns in self._cues:
            if any(p.search(text) for p in patterns):
                for doc_class in doc_classes:
                    if doc_class not in hints:
                        hints.append(doc_class)
        return tuple(hints)

    def has_scheme_token(self, query: str) -> bool:
        """True when the query names a scheme or an unqualified HDFC reference."""
        text = _collapse(query)
        if not text:
            return False
        if any(p.search(text) for p in self._multi):
            return True
        return any(
            any(p.search(text) for p in patterns) for patterns in self._aliases.values()
        )

    def rewrite(self, query: str) -> RewrittenQuery:
        return RewrittenQuery(
            text=(query or "").strip(),
            scheme_keys=self.resolve_scheme_keys(query),
            doc_class_hints=self.doc_class_hints(query),
        )
