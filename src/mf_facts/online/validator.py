"""Node [19] - output validator (architecture.md 5.18 and 7.8, PRD 9.3, D7).

The final gate. No LLM output reaches the user without passing all eight checks
(rule GR9), and any failure produces a safe response rather than the raw text
(rule GR10).

Three properties this module is built to have, in this order:

1. **Pure functions of ``(response, context)``.** No I/O, no LLM, no clock, no
   randomness. That is what makes the acceptance criteria in PRD 13 mechanically
   verifiable, and it is why the module does not import ``generator`` - the
   independence is the point (implementation.md P4 pitfalls). ``tests/
   test_validator.py`` runs every check with a hand-built failing input and no
   model in the process.

2. **The eight checks in the fixed order of architecture.md 5.18.** Checks 5-7
   run last, on the final string, so a repair step cannot reintroduce a violation
   that an earlier check already removed.

3. **Every failure is visible.** ``fired`` and ``reasons`` are returned and
   logged, because a validator that fires silently is worse than no validator -
   it looks like a pass rate problem instead of a defect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from ..common.models import Passage, ValidationResult
from ..common.pii_patterns import find_pii
from ..common.text import sentences, truncate_sentences
from .prompts import FENCE_MARKERS

# architecture.md 5.18 check 5: performance numerics.
_RETURN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*%|\b(?:cagr|xirr|irr|annualized|annualised)\b",
    re.IGNORECASE,
)
# architecture.md 5.18 check 6: recommendation lexicon.
_BUY_SELL = re.compile(
    r"\b(?:should\s+buy|should\s+sell|recommend(?:ed|s)?|best\s+(?:fund|scheme|option)|"
    r"allocate\s+to|invest\s+in|go\s+for|must\s+own|pick\s+the)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s)\]>\"']+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_STAMP = re.compile(r"^Last updated from sources:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

ANSWER_CLASSES = ("factual_scheme", "how_to")
CHECK_IDS = (
    "class_consistency",
    "sentence_count",
    "single_citation",
    "last_updated_stamp",
    "no_performance_numeric",
    "no_recommendation",
    "no_pii_echo",
    "no_prompt_leak",
)

#: The eight of architecture 5.18, in the order it fixes them.
ARCHITECTURE_CHECK_IDS = CHECK_IDS

#: Plus the documented ninth that closes the 5.16/5.18 gap behind P4-T9.
ALL_CHECK_IDS = CHECK_IDS + ("numbers_grounded",)


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Everything a check may read besides the response text.

    §5.18 gives checks 5, 6 and 7 three *different* replacements: a factsheet
    pointer, a refusal, and the PII notice. So the caller supplies all three
    rather than one ``safe_response``; they are templated upstream (rule GR8),
    which is why the validator substitutes text instead of composing its own.

    ``candidate_urls`` is the retrieved candidates' ``source_url`` set. It is
    used as a defensive assertion on ``citation_url``: a citation outside that
    set would mean the citation was built from something other than a retrieved
    chunk, which is the one thing rule GR5 rules out.
    """

    query_class: str
    safe_response: str = ""
    pii_response: str = ""
    performance_response: str = ""
    citation_url: str = ""
    citation_label: str = ""
    candidate_urls: frozenset[str] = frozenset()
    last_updated: str = ""
    passages: tuple[Passage, ...] = ()


def _mangles_number(original: str, truncated: str) -> bool:
    """Did truncation *cut a number in half*?

    The comparison runs truncated-minus-original, not the other way round.
    Getting that backwards makes the check fire whenever truncation drops a
    number-bearing sentence, which is ordinary shortening rather than damage -
    a 4-sentence answer whose 4th sentence mentions a fee would have been refused
    for "mangling" a number that was simply no longer being quoted.

    A number in the truncated text that is absent from the original means a
    numeric token was severed, e.g. "1.03" becoming "1.". Sentence-boundary
    splitting cannot do that today, so this is a guard on
    :func:`common.text.truncate_sentences` rather than a live path - which is
    exactly why it needs its own test.
    """
    original_numbers = set(re.findall(r"\d[\d,.]*", original))
    truncated_numbers = set(re.findall(r"\d[\d,.]*", truncated))
    return bool(truncated_numbers - original_numbers)


def _strip_all_urls(text: str) -> str:
    """Remove every URL, including markdown-link form, keeping the link text out.

    A markdown citation is removed whole rather than unwrapped to its label,
    because "[Source: Groww](https://...)" leaving behind "Source: Groww" would
    read as a fragment of prose in the rebuilt answer.
    """
    text = _MD_LINK.sub("", text)
    return _URL.sub("", text).strip()


def _prose_only(text: str) -> str:
    """The answer body, with the citation and the stamp removed.

    Check 2 counts sentences in prose. The citation and the stamp are appended by
    the caller after generation (architecture.md 10.2), so counting them would
    make a compliant one-sentence answer look like two, and a compliant two-sentence
    answer get truncated for a line the model never wrote.
    """
    stripped = _MD_LINK.sub("", text)
    stripped = _URL.sub("", stripped)
    return _STAMP.sub("", stripped).strip()


def _trailing_lines(text: str) -> str:
    """The citation and stamp lines, so a truncation does not drop them.

    Sentence truncation rewrites the prose, so the appended lines have to be
    re-appended. They are identified structurally (a line carrying a URL, or a
    stamp) rather than by position, because the model may have emitted its own
    URL before the citation.
    """
    kept = [
        line
        for line in (text or "").splitlines()
        if _URL.search(line) or _STAMP.search(line)
    ]
    return ("\n" + "\n".join(kept)) if kept else ""


# --------------------------------------------------------------------------
# The eight checks. Each is a pure function of (text, context) -> (ok, text, reason)
# --------------------------------------------------------------------------


def check_class_consistency(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    """Check 1. A non-answer class may not carry an answer.

    The replacement is class-aware, and that is not decoration. Check 5 only
    fires for ``performance``, and a ``performance`` class always fails *this*
    check first - so if both substituted the generic refusal, check 5 could never
    decide the outcome and the factsheet pointer that 5.18 asks for would never
    be shown. Picking the fallback here is what makes 5.18 checks 1 and 5
    consistent instead of contradictory.
    """
    if context.query_class in ANSWER_CLASSES:
        return True, text, ""
    if context.query_class == "performance":
        return False, context.performance_response or context.safe_response or text, (
            "performance class must not carry an answer"
        )
    return False, context.safe_response or text, (
        f"class {context.query_class!r} must not carry an answer"
    )


def check_sentence_count(
    text: str, context: ValidationContext, limit: int = 3
) -> tuple[bool, str, str]:
    body = _prose_only(text)
    found = sentences(body)
    if len(found) <= limit:
        return True, text, ""
    truncated = truncate_sentences(body, limit)
    if _mangles_number(body, truncated):
        return False, context.safe_response or text, (
            f"more than {limit} sentences and truncation would cut a number"
        )
    # Re-attach the trailing lines the truncation dropped, in their original order.
    trailing = _trailing_lines(text)
    return False, f"{truncated}{trailing}", (
        f"truncated to {limit} sentences from {len(found)}"
    )


def check_single_citation(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    """Exactly one URL on the answer, and it is the structurally chosen one.

    ``context.citation_url`` was built by the caller from ``passages[0].metadata``
    after generation. Any URL the model emitted is discarded unconditionally
    (architecture.md 10.2) - this check is what makes "exactly one link" a
    guarantee rather than an instruction in a prompt.
    """
    if not context.citation_url:
        # Refusal path: no citation is expected, and none is invented here.
        return True, text, ""

    # §5.18: the URL must be a member of the retrieved candidates' source_url
    # set. citation_url is built from passages[0].metadata, so this should hold by
    # construction; asserting it means a future change that builds the citation
    # from somewhere else fails here instead of shipping an unciteable answer.
    if context.candidate_urls and context.citation_url not in context.candidate_urls:
        return False, context.safe_response or text, (
            f"citation {context.citation_url!r} is not a retrieved candidate's source_url"
        )

    found = [url.rstrip(".,);") for url in _URL.findall(text)]
    ours = [url for url in found if url == context.citation_url]
    extra = [url for url in found if url != context.citation_url]

    citation = (
        f"[Source: {context.citation_label}]({context.citation_url})"
        if context.citation_label
        else f"[Source: {context.citation_url}]({context.citation_url})"
    )

    if len(ours) == 1 and not extra:
        return True, text, ""

    body = _strip_all_urls(text)
    rebuilt = f"{body.rstrip()}\n{citation}".strip() if body else citation

    if extra:
        return False, rebuilt, (
            f"discarded {len(extra)} model-emitted URL(s): {extra[:2]}"
        )
    if ours:
        return False, rebuilt, f"citation appeared {len(ours)} times; kept one"
    return False, rebuilt, "citation was missing; re-attached from chunk metadata"


def check_last_updated_stamp(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    if not context.last_updated:
        return True, text, ""
    match = _STAMP.search(text)
    if match and match.group(1) == context.last_updated:
        return True, text, ""
    if not _ISO_DATE.search(context.last_updated):
        return True, text, f"metadata last_updated {context.last_updated!r} is not ISO"
    if match:
        return False, _STAMP.sub(
            f"Last updated from sources: {context.last_updated}", text
        ).strip(), "stamp date did not match chunk metadata; corrected"
    return False, f"{text.rstrip()}\nLast updated from sources: {context.last_updated}".strip(), (
        "stamp was missing; appended from chunk metadata"
    )


def check_no_performance_numeric(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    """Check 5. §5.18: replace with the *factsheet pointer*, not a generic refusal."""
    if context.query_class != "performance":
        return True, text, ""
    hits = _RETURN_PATTERN.findall(text)
    if not hits:
        return True, text, ""
    fallback = context.performance_response or context.safe_response or text
    return False, fallback, f"performance numerics present: {hits[:3]}"


def check_no_recommendation(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    """Check 6. §5.18: replace with a refusal."""
    hits = _BUY_SELL.findall(text)
    if not hits:
        return True, text, ""
    return False, context.safe_response or text, f"recommendation lexicon present: {hits[:3]}"


def check_no_pii_echo(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    """Check 7. §5.18: redact and return the *PII notice*.

    The PII notice rather than a generic refusal, because the model has just
    emitted something that looks like a PAN or an email - which means the answer
    text is not merely ungrounded, it is about to leak. The user is told to
    resubmit without the identifier.
    """
    hits = [(hit.category, text[hit.start : hit.end]) for hit in find_pii(text)]
    if not hits:
        return True, text, ""
    fallback = context.pii_response or context.safe_response or text
    return False, fallback, f"PII pattern echoed in output: {hits[:2]}"


def check_no_prompt_leak(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    hits = [marker for marker in FENCE_MARKERS if marker in text]
    if not hits:
        system_leak = "You are a mutual fund FACTS assistant" in text
        if not system_leak:
            return True, text, ""
        return False, _strip_markers(text), "system prompt text echoed in output"
    return False, _strip_markers(text), f"fence/system markers leaked: {hits}"


def _strip_markers(text: str) -> str:
    stripped = text
    for marker in FENCE_MARKERS:
        stripped = stripped.replace(marker, "")
    return stripped.strip()


def check_numbers_grounded(text: str, context: ValidationContext) -> tuple[bool, str, str]:
    """Every number on the answer appears verbatim in a retrieved passage.

    WHY THIS EXISTS, since architecture 5.18 does not list it
    ------------------------------------------------------
    5.18 fixes eight checks, and this is a ninth. It is here because the rest of
    the spec requires it and nothing else catches it:

    * Prompt rule 2 ("never add a number that does not appear verbatim in a
      PASSAGE") is an instruction, not an enforcement.
    * 5.16 grounding is explicitly a *pre-generation* gate - it compares the
      query to the passages, so it cannot see output at all.
    * Check 5 only guards the ``performance`` class, and check 4's date comes
      from metadata rather than the model.

    So a model that answered "the expense ratio is 0.58%" against a passage
    saying 1.03% would pass all eight checks and ship. implementation.md P4-T9
    (``test_no_number_not_in_passage`` - "a model inventing a number is caught")
    and the P4 acceptance line "0 unsupported numbers in any generated answer"
    both assume a mechanism, and this is it.

    It runs *after* the eight, so their fixed order is untouched, and it fails to
    the same safe response as a failed grounding gate - refusing is the direction
    this product is allowed to be wrong in.
    """
    if not context.passages:
        return True, text, ""
    body = _prose_only(text)
    supported: set[str] = set()
    for passage in context.passages:
        supported.update(_NUMBER.findall(passage.text))
    unsupported = sorted(set(_NUMBER.findall(body)) - supported)
    if not unsupported:
        return True, text, ""
    return False, context.safe_response or text, (
        f"numbers not present in any retrieved passage: {unsupported[:3]}"
    )


#: Checks whose failure *substitutes* a templated response. Once one of these
#: fires the gate is closed: the user sees the refusal, and the answer is gone.
SUBSTITUTING_CHECKS = frozenset(
    {
        "class_consistency",
        "no_performance_numeric",
        "no_recommendation",
        "no_pii_echo",
        "numbers_grounded",
    }
)

#: Checks that only make sense for an answer. After a substitution these are
#: skipped, because there is no answer left for them to have an opinion about -
#: and running them would re-attach a citation and a freshness stamp to a
#: refusal, which is how a "check 3 fired" line and a sourced-looking refusal
#: both end up in the eval report.
ANSWER_ONLY_CHECKS = frozenset(
    {"sentence_count", "single_citation", "last_updated_stamp", "no_prompt_leak"}
)


def _is_substitution(text: str, context: ValidationContext) -> bool:
    """Did this check return one of the caller's templated fallbacks?"""
    return bool(text) and text in {
        context.safe_response,
        context.pii_response,
        context.performance_response,
    }


# Checks 1-4 repair where they can; 5-7 replace with the templated safe response;
# 8 strips. check_numbers_grounded is the documented ninth, run last.
_CHECKS: tuple[tuple[str, Callable[[str, ValidationContext], tuple[bool, str, str]]], ...] = (
    ("class_consistency", check_class_consistency),
    ("sentence_count", check_sentence_count),
    ("single_citation", check_single_citation),
    ("last_updated_stamp", check_last_updated_stamp),
    ("no_performance_numeric", check_no_performance_numeric),
    ("no_recommendation", check_no_recommendation),
    ("no_pii_echo", check_no_pii_echo),
    ("no_prompt_leak", check_no_prompt_leak),
    ("numbers_grounded", check_numbers_grounded),
)


def validate(text: str, context: ValidationContext) -> ValidationResult:
    """Run the checks in order. Returns the text that may be shown.

    A substitution is **terminal**: the first check to replace the text with one
    of the caller's templated fallbacks decides the response, and nothing after it
    may touch that text. Two things go wrong otherwise, and both were observed
    before this rule existed:

    * ``single_citation`` and ``last_updated_stamp`` re-attached a source and a
      freshness date to what had become a refusal, so a refusal looked sourced.
    * ``no_performance_numeric`` never fired. Check 1 had already replaced the
      text, so by check 5 the "12% and a CAGR of 14%" was gone - a performance
      numeric silently stopped being detected. The evidence for a safety check
      cannot live in a string a cosmetic check is still allowed to rewrite.

    So after a substitution the remaining *substituting* checks are still
    evaluated, against the answer that was rejected, and their results are
    reported - the eval report needs to know a return figure was present even
    though the user saw a refusal. The answer-only checks are skipped.

    The returned ``text`` is what the caller should display. On any failure it is
    a repaired string or ``context.safe_response`` - never the untouched model
    output, which is not carried on the result at all (rule GR10).
    """
    current = text or ""
    rejected = text or ""  # the answer as it stands, kept for post-substitution checks
    terminal = False
    fired: list[str] = []
    reasons: list[str] = []

    for check_id, check in _CHECKS:
        if terminal and check_id in ANSWER_ONLY_CHECKS:
            continue

        ok, repaired, reason = check(current if not terminal else rejected, context)

        if not ok:
            fired.append(check_id)
            if reason:
                reasons.append(f"{check_id}: {reason}")

        if not terminal:
            if _is_substitution(repaired, context) and not ok:
                rejected = current  # the answer that was rejected, still worth judging
                current = repaired
                terminal = True
            else:
                current = repaired
                rejected = repaired

    return ValidationResult(
        passed=not fired,
        text=current.strip(),
        safe_response=context.safe_response or current.strip(),
        fired=tuple(fired),
        reasons=tuple(reasons),
    )


def passage_urls(passages: Sequence[Passage]) -> frozenset[str]:
    return frozenset(passage.source_url for passage in passages if passage.source_url)
