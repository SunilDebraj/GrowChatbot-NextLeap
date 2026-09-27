"""Node [10] - the online pipeline (PRD FR2), safety through to a validated answer.

Wires the full path::

    [11] pii -> [12] classifier -> [13] refusal -> [14] rewriter
           -> [15] retriever -> [16] reranker -> [17] grounding
           -> [18] generator -> attach -> [19] validator

Every route ends in one of exactly three things: a templated refusal, a validated
answer built from chunk metadata, or a ``grounding_fail`` refusal. There is no
fourth outcome and no default. A missing LLM, a failed grounding gate, an
unanswerable question and a model that returns ``NO_ANSWER`` all resolve to a
refusal, because the failure mode of the alternative is a confident invented
expense ratio.

Three properties are structural rather than requested in a prompt:

* **PII is terminal.** When the scanner fires, this function returns before the
  rewriter, the classifier, the retriever or any model is given anything. The
  response carries only the boolean and the category names.
* **The citation is chosen from chunk metadata.** ``passages[0].source_url`` and
  ``passages[0].last_updated`` after fusion, attached after generation, and every
  model-emitted URL is discarded by validator check 3 (rules GR5, D4).
* **The validator is the last thing to touch the text.** Its output is what
  reaches the user, and on failure that output is a safe response, never the raw
  model text (rules GR9, GR10).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.config import AppConfig, load_config
from ..common.logging import log_event
from ..common.models import AskResponse, PiiResult, REFUSAL_CLASSES
from ..pipeline.embedder import Embedder
from ..pipeline.sources import load_sources
from ..pipeline.store import ChromaStore
from .classifier import Classifier, LLMClient
from .generator import Generator, build_llm
from .grounding import Grounding
from .pii import PiiScanner
from .refusal import RefusalComposer
from .reranker import Reranker
from .retriever import Retriever
from .rewriter import AliasMap, QueryRewriter
from .validator import ValidationContext, validate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

NOT_YET_IMPLEMENTED = "NOT_YET_IMPLEMENTED"
FOOTER = "Facts-only. No investment advice."

#: Distinguishes "llm was not passed" from "llm is explicitly None". See
#: ``AnswerPipeline.build``.
_UNSET: Any = object()


def _query_hash(query: str) -> str:
    """Stable, non-reversible id for logs. Lets one user's queries be correlated
    across requests without the text itself ever being stored."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class AnswerPipeline:
    """The online pipeline, [11] through [19]."""

    config: AppConfig
    pii_scanner: Any
    classifier: Classifier
    rewriter: QueryRewriter
    refusal: RefusalComposer
    llm: LLMClient | None = None
    retriever: Retriever | None = None
    reranker: Reranker | None = None
    grounding: Grounding | None = None
    generator: Generator | None = None
    scheme_names: dict[str, str] | None = None
    log: Any = None

    @classmethod
    def build(
        cls,
        config: AppConfig | None = None,
        llm: Any = _UNSET,
        root: Path | None = None,
        log: Any = None,
        allow_fake: bool = False,
    ) -> AnswerPipeline:
        """Assemble the pipeline.

        ``llm`` has three states, and collapsing any two of them is a bug:

        * **omitted** - resolve from config. This is the app path, and it is the
          only state that may construct a real, billable provider client.
        * **an object** - use it verbatim. What every test but one uses.
        * **explicitly ``None``** - *no* LLM, so the residual classifier cannot
          run and the answer path fails closed. ``tests/test_online_p3.py``
          relies on this.

        This needed a sentinel rather than a plain ``None`` default because
        "no LLM" and "not specified" were the same value. That was invisible
        while OQ2 was open, since ``build_llm`` returned ``None`` for an
        undecided provider and the fall-through happened to land on fail-closed
        anyway. The moment a real provider was configured, every test that
        passed ``llm=None`` silently started making live, billed API calls and
        asserting against whatever the model happened to say. A test suite that
        can be turned into a network client by a config edit is not a test
        suite.
        """
        base = Path(root) if root else config.root if config else REPO_ROOT
        cfg = config or load_config(base / "config" / "config.yaml")

        rewriter = QueryRewriter(AliasMap.load(base / "config" / "aliases.yaml"))
        registry = load_sources(base / "config" / "sources.yaml")
        scheme_urls = {s.scheme_key: s.url for s in registry if s.enabled}
        names = {s.scheme_key: s.scheme_name for s in registry if s.enabled}

        # A caller-supplied llm wins, including an explicit None meaning "no
        # LLM". Only the omitted case resolves from config.
        client = build_llm(cfg, allow_fake=allow_fake) if llm is _UNSET else llm

        store = ChromaStore(
            path=base / "chroma",
            collection_name=cfg.corpus.collection,
            embedding_model=cfg.embedding.model_id,
            embedding_dim=cfg.embedding.embedding_dim,
        )
        embedder = Embedder(
            cfg.embedding, cache_dir=base / cfg.corpus.embedding_cache_dir
        )

        return cls(
            config=cfg,
            pii_scanner=PiiScanner(cfg),
            classifier=Classifier(rewriter=rewriter, llm=client),
            rewriter=rewriter,
            refusal=RefusalComposer.load(
                base / "config" / "educational_links.yaml",
                scheme_urls=scheme_urls,
                scheme_names=names,
                local_note=(base / "assets" / "scope_note.txt").read_text(encoding="utf-8"),
            ),
            llm=client,
            retriever=Retriever(cfg, store, embedder),
            reranker=Reranker(cfg),
            grounding=Grounding(cfg),
            generator=Generator(cfg, llm=client),
            scheme_names=names,
            log=log,
        )

    # -- the pipeline -------------------------------------------------------

    def ask(self, query: str) -> AskResponse:
        pii = self.pii_scanner.scan(query)

        if pii.is_pii:
            # Nothing downstream sees the query. Not the rewriter, not the
            # classifier, not the retriever, not a model.
            response = self.refusal.compose("pii", rule_id="pii:detected")
            return self._finalize(
                AskResponse(
                    query_class="pii",
                    route="refusal",
                    text=response.text,
                    rule_id="pii:detected",
                    pii_detected=True,
                    pii_categories=pii.categories,
                ),
                query,
            )

        classification = self.classifier.classify(pii.sanitized_query, pii)
        query_class = classification.query_class

        if query_class in REFUSAL_CLASSES:
            return self._refuse(query, query_class, classification.rule_id, pii, classification.confidence)

        return self._answer(query, pii, classification, query_class)

    def _refuse(
        self,
        query: str,
        query_class: str,
        rule_id: str,
        pii: PiiResult,
        confidence: float = 0.0,
    ) -> AskResponse:
        scheme_key = None
        multi_scheme = False
        if query_class in ("performance", "grounding_fail"):
            keys = self.rewriter.resolve_scheme_keys(pii.sanitized_query)
            # D4 permits exactly one link, so a multi-scheme refusal links the
            # first scheme named and says so, rather than linking nothing.
            scheme_key = keys[0] if keys else None
            multi_scheme = len(keys) > 1
        refusal = self.refusal.compose(
            query_class,
            scheme_key=scheme_key,
            rule_id=rule_id,
            multi_scheme=multi_scheme,
        )
        return self._finalize(
            AskResponse(
                query_class=query_class,
                route="refusal",
                text=refusal.text,
                link_label=refusal.link_label,
                link_url=refusal.link_url,
                rule_id=rule_id,
                confidence=confidence,
            ),
            query,
        )

    def _answer(
        self,
        query: str,
        pii: PiiResult,
        classification: Any,
        query_class: str,
    ) -> AskResponse:
        """rewrite -> retrieve -> rerank -> ground -> generate -> attach -> validate."""
        assert self.retriever and self.reranker and self.grounding and self.generator

        rewritten = self.rewriter.rewrite(pii.sanitized_query)
        hints = rewritten.doc_class_hints

        if not rewritten.scheme_keys:
            return self._refuse(query, "out_of_scope", "ground:no_scheme_key", pii,
                                classification.confidence)

        candidates = self.retriever.retrieve(
            self.retriever.embed_query(rewritten.text), rewritten.scheme_keys
        )
        passages = self.reranker.rerank(
            candidates,
            query=rewritten.text,
            doc_class_hints=hints,
            numeric_question=Grounding.is_numeric_question(rewritten.text),
        )

        grounding = self.grounding.check(rewritten.text, passages, hints)
        if not grounding.passed:
            log_event(
                _logger(),
                "grounding.failed",
                query_hash=_query_hash(query),
                route="refusal",
                **{"class": query_class},
                rule_id="ground:failed",
                grounding_score=grounding.score,
                status=grounding.reason,
                scheme_key=rewritten.scheme_keys[0] if rewritten.scheme_keys else "",
            )
            return self._refuse(
                query, "grounding_fail", "ground:failed", pii, classification.confidence
            )

        names = [
            (self.scheme_names or {}).get(key, key) for key in rewritten.scheme_keys
        ]
        generation = self.generator.generate(rewritten.text, passages, names)

        if generation.is_no_answer or not generation.text.strip():
            # NO_ANSWER is a refusal, not an error: the model was asked for it
            # precisely so that "not in the corpus" has a defined outcome.
            return self._refuse(
                query, "grounding_fail", "ground:no_answer", pii, classification.confidence
            )

        # P4-T6: the citation and the stamp come from chunk metadata, attached
        # after generation. passages[0] is the fused top passage.
        best = passages[0]
        citation_url = best.source_url
        last_updated = best.last_updated
        label = str(best.metadata.get("scheme_name") or best.scheme_key)

        attached = self._attach(generation.text, citation_url, label, last_updated)

        # §5.18 gives checks 5, 6 and 7 three different replacements, so all
        # three are composed here, where the templates live, rather than inside
        # the validator. Checks 5 and 7 are unreachable in practice - performance
        # is terminal in P3 and PII never reaches generation - but they must be
        # correct if either route ever changes.
        safe = self.refusal.compose(
            "grounding_fail", scheme_key=rewritten.scheme_keys[0], rule_id="validator:safe"
        ).text
        pii_notice = self.refusal.compose("pii", rule_id="pii:detected").text
        factsheet = self.refusal.compose(
            "performance", scheme_key=rewritten.scheme_keys[0], rule_id="validator:performance"
        ).text

        result = validate(
            attached,
            ValidationContext(
                query_class=query_class,
                safe_response=safe,
                pii_response=pii_notice,
                performance_response=factsheet,
                citation_url=citation_url,
                citation_label=label,
                candidate_urls=self.retriever.candidate_source_urls(candidates),
                last_updated=last_updated,
                passages=tuple(passages),
            ),
        )

        # §5.18: every fail is logged with check_id and reason. A validator that
        # fires silently is worse than no validator, because it looks like a
        # retrieval quality problem instead of a defect.
        for check_id, reason in zip(result.fired, result.reasons):
            log_event(
                _logger(),
                "validator.failed",
                query_hash=_query_hash(query),
                route="safe_response",
                rule_id=check_id,
                status=reason,
            )

        log_event(
            _logger(),
            "answer.validated",
            query_hash=_query_hash(query),
            route="answer" if result.passed else "safe_response",
            **{"class": query_class},
            rule_id=classification.rule_id,
            status="pass" if result.passed else "fail",
            grounding_score=grounding.score,
            validator_flags=list(result.fired),
            chunk_count=len(passages),
            scheme_key=rewritten.scheme_keys[0] if rewritten.scheme_keys else "",
            doc_class=",".join(hints),
            no_answer=generation.is_no_answer,
        )

        return self._finalize(
            AskResponse(
                query_class=query_class,
                route="answer" if result.passed else "safe_response",
                text=result.text,
                link_label=label,
                link_url=citation_url,
                rule_id=classification.rule_id,
                confidence=classification.confidence,
                scheme_keys=rewritten.scheme_keys,
                doc_class_hints=hints,
                citation_url=citation_url,
                last_updated=last_updated,
                grounding_passed=grounding.passed,
                grounding_score=grounding.score,
                validator_checks=result.fired,
                answer_model=generation.model,
                retrieved_ids=tuple(passage.id for passage in passages),
            ),
            query,
        )

    @staticmethod
    def _attach(text: str, citation_url: str, label: str, last_updated: str) -> str:
        """Append the citation and the freshness stamp, in that order."""
        parts = [text.strip()]
        if citation_url:
            parts.append(f"[Source: {label}]({citation_url})")
        if last_updated:
            parts.append(f"Last updated from sources: {last_updated}")
        return "\n".join(part for part in parts if part)

    # -- observability ------------------------------------------------------

    def _finalize(self, response: AskResponse, query: str) -> AskResponse:
        """Log the audit fields and nothing else.

        Architecture 14.2: the log carries query_hash, route, class and rule_id -
        never the query text. ``pii_categories`` is written for a PII request even
        when ``log_raw_queries`` is on, because the flag does not get to unblock
        logging an identifier.
        """
        if self.log is None:
            return response
        fields: dict[str, Any] = {
            "query_hash": _query_hash(query),
            "route": response.route,
            "class": response.query_class,
            "rule_id": response.rule_id,
        }
        if response.pii_detected:
            fields["pii_detected"] = True
            fields["pii_categories"] = list(response.pii_categories)
        self.log.info("query", extra=fields)
        return response


def _logger():
    import logging

    return logging.getLogger("mf_facts.online.ask")
