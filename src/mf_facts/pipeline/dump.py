"""Corpus dump: chunks and vectors as text (read-only inspection).

Written because the corpus is otherwise invisible. Raw fetches land in
``raw_cache/`` under content-addressed digests, and the chunks themselves live
only inside Chroma as vectors plus text. Neither is readable in an editor, so
there is no way to check what the chunker actually produced without writing
throwaway code that reaches into Chroma directly.

    python -m mf_facts.pipeline.dump              # both files
    python -m mf_facts.pipeline.dump --what chunks
    python -m mf_facts.pipeline.dump --out-dir /tmp/corpus --no-vectors

Writes ``artifacts/chunks.txt`` and ``artifacts/embeddings.txt``.

Two deliberate choices, both about keeping the dump trustworthy rather than
merely pretty:

* It reads Chroma, not the embedding cache. Chroma is what retrieval will
  actually see, so a dump of the cache could disagree with the store and send
  someone debugging the wrong layer. Chunk order is sorted by
  (scheme_key, doc_class, chunk_index) so two runs produce byte-identical files.

* Vectors are rounded to 6 decimals. The stored vectors are float64 and
  ``cache/embeddings/*.json`` still has them at full precision, but printing
  them raw puts runs of 12+ digits on the page, and
  ``tests/test_no_pii_at_rest.py`` treats a 12-digit run in ``artifacts/`` as a
  possible PAN. Those runs are vector coordinates, not account numbers, and
  rounding keeps the dump inside the check the repo already runs. 6 decimals is
  well inside the precision that separates these chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.config import AppConfig, load_config
from ..common.errors import CollectionUnavailable, ConfigError
from .store import ChromaStore

CHROMA_DIRNAME = "chroma"
VECTOR_DECIMALS = 6
VECTORS_PER_LINE = 6
RULE = "=" * 78
THIN = "-" * 78

# Metadata keys printed in this order. Anything else the store carries is
# appended after these, so a new field shows up in the dump without a code
# change and without silently disappearing.
_METADATA_ORDER = (
    "scheme_key",
    "scheme_name",
    "doc_class",
    "section",
    "chunk_index",
    "token_count",
    "publisher",
    "effective_date",
    "last_updated",
    "source_url",
    "content_hash",
)

# The one column that is long by nature; the rest are aligned to this width.
_LABEL_WIDTH = 16


def _repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    # src/mf_facts/pipeline/dump.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[3]


def _store(root: Path, config: AppConfig) -> ChromaStore:
    return ChromaStore(
        path=root / CHROMA_DIRNAME,
        collection_name=config.corpus.collection,
        embedding_model=config.embedding.model_id,
        embedding_dim=config.embedding.embedding_dim,
    )


def _sorted_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Chroma's row order is not a contract; make the dump's order ours."""
    # Chroma hands back numpy arrays, so `x or []` raises on truth-testing an
    # array. Index the three parallel columns defensively instead.
    ids = payload.get("ids")
    if ids is None:
        return []
    documents = payload.get("documents")
    metadatas = payload.get("metadatas")
    embeddings = payload.get("embeddings")

    rows: list[dict[str, Any]] = []
    for position, chunk_id in enumerate(ids):
        rows.append(
            {
                "position": position,
                "id": chunk_id,
                "text": documents[position] if documents is not None else None,
                "metadata": (
                    dict(metadatas[position] or {}) if metadatas is not None else {}
                ),
                "vector": embeddings[position] if embeddings is not None else None,
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["metadata"].get("scheme_key", "")),
            str(row["metadata"].get("doc_class", "")),
            _as_int(row["metadata"].get("chunk_index")),
            row["id"],
        )
    )
    return rows


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _header(title: str, config: AppConfig, rows: list[dict[str, Any]], root: Path) -> list[str]:
    manifest: dict[str, Any] = {}
    manifest_path = root / "artifacts" / "corpus_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}

    lines = [
        RULE,
        title,
        RULE,
        f"generated_at     : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"source           : {CHROMA_DIRNAME}/ collection {config.corpus.collection!r}",
        f"chunks           : {len(rows)}",
        f"embedding_model  : {config.embedding.model_id}",
        f"embedding_dim    : {config.embedding.embedding_dim}",
        f"corpus_version   : {manifest.get('corpus_version', '<unknown>')}",
        f"corpus_hash      : {manifest.get('corpus_hash', '<unknown>')}",
        f"built_at         : {manifest.get('built_at', '<unknown>')}",
        f"chunking_strategy: {manifest.get('chunking_strategy', '<unknown>')}",
        "",
    ]
    return lines


def _metadata_block(metadata: dict[str, Any]) -> list[str]:
    ordered = [key for key in _METADATA_ORDER if key in metadata]
    extra = sorted(key for key in metadata if key not in _METADATA_ORDER)
    lines = []
    for key in ordered + extra:
        value = metadata[key]
        if value is None or value == "":
            continue
        lines.append(f"{key:<{_LABEL_WIDTH}} : {value}")
    return lines


def render_chunks(rows: list[dict[str, Any]], config: AppConfig, root: Path) -> str:
    lines = _header("CHUNK DUMP - the text the retriever will search", config, rows, root)
    lines += [
        "Each block below is one Chroma document. The TEXT section is verbatim.",
        "",
    ]
    for number, row in enumerate(rows, start=1):
        text = row["text"] or ""
        lines += [
            RULE,
            f"CHUNK {number} of {len(rows)}",
            RULE,
            f"{'id':<{_LABEL_WIDTH}} : {row['id']}",
        ]
        lines += _metadata_block(row["metadata"])
        lines += [
            f"{'text_chars':<{_LABEL_WIDTH}} : {len(text)}",
            THIN,
            text if text.strip() else "<empty document>",
            "",
        ]
    return "\n".join(lines) + "\n"


def render_vectors(rows: list[dict[str, Any]], config: AppConfig, root: Path) -> str:
    lines = _header("EMBEDDING DUMP - the vectors retrieval will compare", config, rows, root)
    lines += [
        f"Vectors are rounded to {VECTOR_DECIMALS} decimals, {VECTORS_PER_LINE} per line,",
        "dimension index in brackets. Full float64 precision is in",
        f"{config.corpus.embedding_cache_dir}/<digest>.json if you need it.",
        "",
    ]
    for number, row in enumerate(rows, start=1):
        raw = row["vector"]
        vector = [float(value) for value in raw] if raw is not None else []
        lines += [
            RULE,
            f"CHUNK {number} of {len(rows)}",
            RULE,
            f"{'id':<{_LABEL_WIDTH}} : {row['id']}",
        ]
        lines += _metadata_block(row["metadata"])
        lines += [f"{'dims':<{_LABEL_WIDTH}} : {len(vector)}", THIN]

        if not vector:
            lines += ["<no vector stored>"]
        else:
            for start in range(0, len(vector), VECTORS_PER_LINE):
                window = vector[start : start + VECTORS_PER_LINE]
                cells = [
                    f"[{start + offset:>3}] {value:.{VECTOR_DECIMALS}f}"
                    for offset, value in enumerate(window)
                ]
                lines.append("  " + "  ".join(cells))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mf_facts.pipeline.dump",
        description="Write the indexed chunks and their vectors to readable text files.",
    )
    parser.add_argument("--root", help="repository root (default: inferred from this file)")
    parser.add_argument("--out-dir", help="output directory (default: <root>/artifacts)")
    parser.add_argument(
        "--what",
        choices=("all", "chunks", "vectors"),
        default="all",
        help="which file to write (default: all)",
    )
    parser.add_argument(
        "--no-vectors",
        action="store_true",
        help="skip the embedding file; the vector dump is the slow one to skim",
    )
    args = parser.parse_args(argv)

    root = _repo_root(args.root)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "artifacts"

    try:
        config = load_config(root / "config" / "config.yaml")
        store = _store(root, config)
        payload = store.get_all_with_vectors()
    except (ConfigError, CollectionUnavailable) as exc:
        print(f"dump failed: {exc}", file=sys.stderr)
        return 1

    rows = _sorted_rows(payload)
    if not rows:
        print(
            f"dump failed: collection {config.corpus.collection!r} is empty; "
            "run the build pipeline first",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if args.what in ("all", "chunks"):
        target = out_dir / "chunks.txt"
        target.write_text(render_chunks(rows, config, root), encoding="utf-8")
        written.append(target)

    if args.what == "vectors" or (args.what == "all" and not args.no_vectors):
        target = out_dir / "embeddings.txt"
        target.write_text(render_vectors(rows, config, root), encoding="utf-8")
        written.append(target)

    for path in written:
        print(f"wrote {path.relative_to(root) if path.is_relative_to(root) else path} "
              f"({path.stat().st_size:,} bytes)")
    print(f"{len(rows)} chunks from collection {config.corpus.collection!r}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
