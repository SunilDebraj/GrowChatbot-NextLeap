"""Node [18] - generator (architecture.md 5.17, PRD FR2 step 7).

Emits answer **text only**. No URL, no stamp - both are attached by the caller from
chunk metadata (rule GR5), which is what makes "exactly one link" structural
rather than a thing the model was asked to behave.

PRD OQ2 (which provider) is still open, so no provider name appears anywhere in
this module. ``build_llm`` dispatches on ``generation.provider`` from config, and
an unrecognised value is a configuration error rather than a silent fallback. That
matters more here than in P3: in P3 a wrong provider would fail a classifier
refusal, in P4 it would either fail closed or, worse, ship to a user.

    ``FakeLLM`` exists so the whole answer path is provable with no API key, which is
    the handling architecture.md OQ2 and implementation.md 2.1 prescribe while the
    decision is open. It is deliberately not reachable from the app: ``build_llm``
    refuses ``provider: fake`` unless ``allow_fake=True`` is passed explicitly, so it
    cannot reach a demo by accident (implementation.md P4 pitfalls).

    The generation path has no tools, no network and no filesystem access beyond the
    HTTP call to the configured provider (rule GR13). There is nothing here that can
    act on an instruction found in a retrieved page.
    """

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..common.config import AppConfig
from ..common.env import load_env
from ..common.errors import ConfigError, LLMError
from ..common.models import Passage
from ..common.text import truncate_sentences
from .prompts import FENCE_CLOSE, FENCE_OPEN, NO_ANSWER, SYSTEM_PROMPT, build_user_prompt


class LLMClient(Protocol):
    """The provider surface. Tests and ``FakeLLM`` satisfy it structurally.

    Two call shapes reach this one interface, and that is a consequence of the
    classifier being a separate node rather than a shared prompt: the residual
    stage calls ``generate(prompt=..., text=..., json_only=True)``
    (classifier.py, architecture 5.12) while this module calls
    ``generate(system_prompt=..., user_prompt=..., max_tokens=..., model=...)``
    (architecture 10.1). A client has to normalise both - see
    :func:`_messages`. P3 had only the first shape, so this is the first point
    where the two contracts meet.
    """

    def generate(self, **kwargs: Any) -> str: ...


def _messages(kwargs: dict[str, Any]) -> tuple[str, str]:
    """Normalise both call shapes to ``(system, user)``."""
    if "prompt" in kwargs:
        return str(kwargs.get("prompt") or ""), str(kwargs.get("text") or "")
    return str(kwargs.get("system_prompt") or ""), str(kwargs.get("user_prompt") or "")


@dataclass(frozen=True, slots=True)
class Generation:
    """What the model returned, before anything is attached to it."""

    text: str
    model: str
    provider: str
    is_no_answer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "is_no_answer": self.is_no_answer,
        }


class FakeLLM:
    """A deterministic, offline stand-in for a provider.

    It is extractive, not generative: it returns the first three sentences of
    ``passages[0]`` - the fused top passage. That is enough to exercise retrieval,
    grounding, citation attachment, every validator check and the ``NO_ANSWER``
    branch, and it cannot be mistaken for a quality result: the P2 gate already
    records that no model has been evaluated, and OQ2 is still open.

    Using ``passages[0]`` rather than re-ranking the passages here is deliberate,
    and it is a stronger test rather than a weaker one. The reranker has already
    decided which chunk is best (architecture 7.7) and ``ask`` cites
    ``passages[0]``, so copying it means **the answer and the citation always come
    from the same chunk** - which is the D4 property under test. An earlier
    version scored passages by shared question terms and picked the ``overview``
    chunk for "what is the exit load", because the scheme name is in every chunk
    of a scheme and outweighed the actual fact term; that was a bug in the stand-
    in, and it made the fake contradict the citation it was handed.

    ``script`` forces a specific reply, which is how the tests drive the failure
    paths a well-behaved extractive client never takes on its own: emitting three
    URLs, inventing a number, leaking a fence marker, saying NO_ANSWER.
    """

    def __init__(
        self,
        script: str | None = None,
        model: str = "fake-extractive",
        classification: str = "factual_scheme",
        max_sentences: int = 3,
    ) -> None:
        self.script = script
        self.model = model
        self.classification = classification
        self.max_sentences = max_sentences
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        system, user = _messages(kwargs)

        # The residual classifier's shape. It is only ever reached when the
        # rewriter already found a scheme token (classifier.py, the
        # has_scheme_token gate), so factual_scheme is the honest default rather
        # than a guess: the rules that carry regulatory weight have already run.
        if kwargs.get("json_only"):
            return f'{{"label": "{self.classification}", "confidence": 0.9}}'

        if self.script is not None:
            return self.script

        passages = _extract_passage_text(user)
        if not passages:
            return NO_ANSWER
        return truncate_sentences(passages[0].replace("\n", " "), self.max_sentences)


def _extract_passage_text(prompt: str) -> list[str]:
    """Recover the passage texts from a rendered user prompt.

    Parses the fence that :mod:`prompts` wrote rather than being handed the
    passages directly, because ``FakeLLM`` sits behind the same provider
    interface a real client implements and that interface only carries a prompt.
    If the fence format changes, this is what breaks - which is the point: the
    tests should fail rather than silently pass on an empty passage list.
    """
    texts: list[str] = []
    current: list[str] | None = None
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(FENCE_OPEN):
            current = []
            continue
        if stripped.startswith(FENCE_CLOSE):
            if current is not None:
                texts.append("\n".join(current).strip())
            current = None
            continue
        if current is not None:
            current.append(line)
    return texts


class HttpLLMClient:
    """A provider client whose identity comes entirely from config.

    Deliberately minimal and deliberately dumb: it POSTs the prompt and returns
    the text. Anything provider-specific - auth header, response envelope, error
    taxonomy - is a subclass away, and the point of OQ2 being open is that this
    shape does not have to change when the decision is made.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key_env: str = "",
        timeout_s: float = 30.0,
        base_url: str = "",
        user_agent: str = "mf-facts/0.1 (+https://localhost)",
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.base_url = base_url
        self.user_agent = user_agent

    def generate(self, **kwargs: Any) -> str:
        import os
        import urllib.error
        import urllib.request

        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if not key:
            raise LLMError(
                f"provider {self.provider!r} needs ${self.api_key_env} to be set; "
                "OQ2 is unresolved, so this is expected until a provider is chosen"
            )
        if not self.base_url:
            raise ConfigError(
                f"provider {self.provider!r} has no base_url configured; add one "
                "rather than hardcoding a provider URL in code"
            )

        system, user = _messages(kwargs)
        body = json.dumps(
            {
                "model": kwargs.get("model", self.model),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": kwargs.get("temperature", 0.0),
                "max_tokens": kwargs.get("max_tokens", 180),
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                # Not cosmetic. urllib defaults to "Python-urllib/3.x", and
                # edge-fronted providers sit behind bot protection that rejects
                # that signature outright - the live endpoint answered 403
                # Cloudflare "error 1010" to a default-UA request and 200 to this
                # one, with the same valid key. Identify the client honestly.
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # A bare "HTTP Error 403: Forbidden" sends an operator hunting. The
            # body carries the actual reason (bad key, unknown model, bot block),
            # and it never contains the key - the key only ever went out in a
            # request header. Bounded, because a provider can return anything.
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300].strip()
            except Exception:  # noqa: BLE001 - diagnostics must never mask the error
                pass
            raise LLMError(
                f"{self.provider} returned HTTP {exc.code} for model {self.model!r}"
                + (f": {detail}" if detail else "")
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

        try:
            return str(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected {self.provider} response shape: {exc}") from exc


def build_llm(config: AppConfig, allow_fake: bool = False) -> LLMClient | None:
    """Resolve the provider from config. ``None`` means "fail closed" downstream.

    Returning ``None`` for a real-but-unconfigured provider is deliberate: the
    residual classifier and the generator both treat a missing client as a refusal
    rather than as licence to answer from priors.
    """
    provider = (config.generation.provider or "").strip()
    model = (config.generation.model or "").strip()

    if provider in ("", "<TBD: OQ2>"):
        return None
    if provider == "fake":
        if not allow_fake:
            raise ConfigError(
                "generation.provider is 'fake'. FakeLLM must not reach the app path "
                "(implementation.md P4 pitfalls); pass allow_fake=True to use it in a test."
            )
        return FakeLLM(
            model=model or "fake-extractive",
            max_sentences=int(config.generation.max_sentences),
        )
    if provider in ("http", "chat_completions"):
        # A wire shape, not a vendor. Named for the request/response envelope it
        # speaks rather than for any company, because OQ2 is still open and the
        # name should not prejudge it. An openai-compatible endpoint is one
        # instance of this shape, configured via generation.base_url.
        #
        # .env is read here, at the point the key is actually needed, rather than
        # at import: a test that sets the variable itself must win, and nothing
        # else in the project should grow a side effect on import.
        load_env()
        return HttpLLMClient(
            provider=provider,
            model=model,
            api_key_env=config.generation.api_key_env,
            base_url=config.generation.base_url,
            timeout_s=config.generation.timeout_s,
        )
    raise ConfigError(
        f"unknown generation.provider {provider!r}. OQ2 is open; add the provider here "
        "explicitly rather than falling back to a default."
    )


class Generator:
    def __init__(self, config: AppConfig, llm: LLMClient | None = None) -> None:
        self.config = config
        self.llm = llm
        self.temperature = float(config.generation.temperature)
        self.max_tokens = int(config.generation.max_tokens)
        self.model = (config.generation.model or "").strip()
        self.provider = (config.generation.provider or "").strip()

    @property
    def available(self) -> bool:
        return self.llm is not None

    def generate(
        self,
        question: str,
        passages: Sequence[Passage],
        scheme_names: Sequence[str] = (),
    ) -> Generation:
        if self.llm is None:
            # Fail closed. The caller turns this into a grounding_fail refusal.
            return Generation(
                text=NO_ANSWER,
                model=self.model or "none",
                provider=self.provider or "none",
                is_no_answer=True,
            )

        model_name = getattr(self.llm, "model", "") or self.model
        try:
            raw = self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(question, passages, scheme_names),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model=model_name,
            )
        except LLMError:
            raise
        except Exception as exc:  # a provider must not surface as a crash
            raise LLMError(f"generation failed: {exc}") from exc

        text = (raw or "").strip()
        is_no_answer = text.strip().upper() == NO_ANSWER
        return Generation(
            text=text,
            model=str(model_name),
            provider=self.provider or "unknown",
            is_no_answer=is_no_answer,
        )
