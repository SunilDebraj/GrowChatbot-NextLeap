"""[2] Fetcher: retrieve raw bytes per URL, content-addressed cache (architecture.md 5.2).

A fetch failure marks that URL and continues; the build only fails when a scheme
ends with zero successful documents, so a single bad page cannot silently drop a
whole scheme.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..common.logging import log_event
from ..common.models import RawDocument, SourceSpec

USER_AGENT = (
    "Mozilla/5.0 (compatible; mf-facts-bot/0.1; research prototype; "
    "contact: local development)"
)

DEFAULT_TIMEOUT_S = 45.0
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_path(cache_dir: Path, digest: str, content_type: str) -> Path:
    suffix = ".html"
    if "pdf" in content_type.lower():
        suffix = ".pdf"
    elif "json" in content_type.lower():
        suffix = ".json"
    return cache_dir / digest[:2] / f"{digest}{suffix}"


class Fetcher:
    """Retrieves one URL once, then serves every spec that shares it."""

    def __init__(
        self,
        cache_dir: str | Path,
        refresh: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self._owns_client = client is None
        self._client = client or httpx.Client(
            follow_redirects=True,
            timeout=timeout_s,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"},
        )

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch(self, url: str) -> RawDocument:
        """Fetch ``url``, using the cache unless ``refresh`` is set.

        Returns a RawDocument with ``status=0`` and an ``error`` string on
        failure rather than raising, because one unreachable page must not abort
        the build.
        """
        retrieved_at = _now()

        if not self.refresh:
            cached = self._find_cached(url)
            if cached is not None:
                path, digest, content_type = cached
                return RawDocument(
                    source_id="",
                    url=url,
                    resolved_url=url,
                    content=path.read_bytes(),
                    content_type=content_type,
                    status=200,
                    retrieved_at=retrieved_at,
                    from_cache=True,
                )

        last_error = ""
        for attempt in range(2):
            try:
                response = self._client.get(url)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

            if response.status_code in RETRY_STATUSES:
                last_error = f"HTTP {response.status_code}"
                continue
            if response.status_code >= 400:
                return RawDocument(
                    source_id="",
                    url=url,
                    resolved_url=str(response.url),
                    content=b"",
                    content_type=response.headers.get("content-type", ""),
                    status=response.status_code,
                    retrieved_at=retrieved_at,
                    from_cache=False,
                    error=f"HTTP {response.status_code}",
                )

            data = response.content
            content_type = response.headers.get("content-type", "")
            digest = _content_hash(data)
            path = _cache_path(self.cache_dir, digest, content_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

            log_event(
                __import__("logging").getLogger("mf_facts.fetcher"),
                "fetch.ok",
                status=response.status_code,
                count=len(data),
            )

            return RawDocument(
                source_id="",
                url=url,
                resolved_url=str(response.url),
                content=data,
                content_type=content_type,
                status=response.status_code,
                retrieved_at=retrieved_at,
                from_cache=False,
            )

        return RawDocument(
            source_id="",
            url=url,
            resolved_url=url,
            content=b"",
            content_type="",
            status=0,
            retrieved_at=retrieved_at,
            from_cache=False,
            error=last_error or "unknown fetch failure",
        )

    def _find_cached(self, url: str) -> tuple[Path, str, str] | None:
        """Locate a cached body for ``url``.

        The cache is content-addressed, so a URL lookup needs a small side index
        mapping url -> digest. The index is advisory: if it is missing or stale
        the caller simply refetches.
        """
        index_path = self.cache_dir / "index.json"
        if not index_path.is_file():
            return None
        try:
            import json

            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

        entry = index.get(url)
        if not entry:
            return None
        digest = entry.get("digest")
        if not digest:
            return None
        for suffix in (".html", ".pdf", ".json"):
            candidate = self.cache_dir / digest[:2] / f"{digest}{suffix}"
            if candidate.is_file():
                return candidate, digest, entry.get("content_type", "")
        return None

    def record_in_cache_index(self, url: str, document: RawDocument) -> None:
        """Add ``url -> digest`` to the advisory cache index."""
        if not document.content:
            return
        import json

        digest = _content_hash(document.content)
        index_path = self.cache_dir / "index.json"
        index: dict = {}
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                index = {}
        index[url] = {"digest": digest, "content_type": document.content_type}
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def enabled_specs(specs: list[SourceSpec]) -> list[SourceSpec]:
    return [spec for spec in specs if spec.enabled]
