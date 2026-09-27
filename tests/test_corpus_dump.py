"""Contract tests for the corpus dump (pipeline/dump.py).

The dump exists for human inspection, so these tests protect the properties a
person relies on: every chunk appears, ids line up with vectors, and two runs
agree. A dump that silently dropped the last chunk or shuffled rows would be
worse than no dump, because it looks authoritative.
"""

from __future__ import annotations

import pytest

from mf_facts.pipeline import dump
from mf_facts.pipeline.store import ChromaStore


@pytest.fixture(scope="module")
def corpus(request):
    """The real built corpus. Skips when it has not been built yet."""
    root = request.config.rootpath
    if not (root / "chroma").exists():
        pytest.skip("chroma/ does not exist; run the build pipeline first")
    store = ChromaStore(
        path=root / "chroma",
        collection_name="mf_facts_v1",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    )
    if store.count() == 0:
        pytest.skip("collection mf_facts_v1 is empty")
    return root, store


def test_every_stored_chunk_appears_in_the_chunks_file(corpus):
    root, store = corpus
    payload = store.get_all_with_vectors()
    rows = dump._sorted_rows(payload)

    assert len(rows) == store.count()
    assert len(rows) == 20, "P1 corpus is 20 chunks; a change here needs a reason"

    rendered = dump.render_chunks(rows, dump.load_config(root / "config" / "config.yaml"), root)
    for row in rows:
        assert row["id"] in rendered
        assert (row["text"] or "").strip() in rendered


def test_vector_ids_and_dimensions_match_the_store(corpus):
    root, store = corpus
    config = dump.load_config(root / "config" / "config.yaml")
    payload = store.get_all_with_vectors()
    rows = dump._sorted_rows(payload)

    assert all(len(row["vector"]) == config.embedding.embedding_dim for row in rows)
    assert all(len(row["vector"]) == 384 for row in rows)

    rendered = dump.render_vectors(rows, config, root)
    for row in rows:
        assert row["id"] in rendered


def test_chunk_and_vector_dumps_describe_the_same_chunks(corpus):
    root, store = corpus
    config = dump.load_config(root / "config" / "config.yaml")
    rows = dump._sorted_rows(store.get_all_with_vectors())

    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate chunk id in the dump"
    # Both renderers iterate one shared, deterministically sorted row list, so a
    # mismatch here means one of them started sorting independently.
    assert ids == [row["id"] for row in dump._sorted_rows(store.get_all_with_vectors())]


def test_dump_row_order_is_deterministic(corpus):
    root, store = corpus
    rows = dump._sorted_rows(store.get_all_with_vectors())
    shuffled = {key: value for key, value in reversed(list(rows[0].items()))}
    assert rows[0]["id"] == shuffled["id"]
    assert rows == sorted(
        rows,
        key=lambda row: (
            str(row["metadata"].get("scheme_key", "")),
            str(row["metadata"].get("doc_class", "")),
            dump._as_int(row["metadata"].get("chunk_index")),
            row["id"],
        ),
    )


def test_vector_precision_keeps_the_dump_free_of_account_sized_digit_runs(corpus):
    """The reason for VECTOR_DECIMALS, asserted so it is not quietly removed.

    `account:long` in common/pii_patterns.py is [0-9]{12,}, and full float64
    vectors produce digit runs that long from coordinates rather than account
    numbers. tests/test_no_pii_at_rest.py already covers artifacts/ as a whole,
    but it scrubs 32+ char hex tokens first, so a corpus SHA256 in the dump
    header cannot mask a genuine regression here. This checks the vector lines
    on their own, where the header is not in scope.

    If this fails, raise the sweep's scope deliberately - do not bump
    VECTOR_DECIMALS back up.
    """
    import re

    root, store = corpus
    config = dump.load_config(root / "config" / "config.yaml")
    rows = dump._sorted_rows(store.get_all_with_vectors())

    rendered = dump.render_vectors(rows, config, root)
    # Only lines carrying dimension markers, i.e. actual vector values.
    vector_lines = [line for line in rendered.splitlines() if re.search(r"\[\s*\d+\]", line)]
    assert len(vector_lines) == sum(
        -(-len(row["vector"]) // dump.VECTORS_PER_LINE) for row in rows
    ), "not every vector line was selected; the filter drifted from the format"

    long_runs = [run for line in vector_lines for run in re.findall(r"\d{12,}", line)]
    assert not long_runs, f"vector coordinates long enough to look like an account: {long_runs[:3]}"


def test_chunk_dump_carries_the_stored_text_verbatim(corpus):
    """The dump's whole purpose is showing the real chunk text, not a summary."""
    root, store = corpus
    config = dump.load_config(root / "config" / "config.yaml")
    rows = dump._sorted_rows(store.get_all_with_vectors())
    rendered = dump.render_chunks(rows, config, root)

    for row in rows:
        text = (row["text"] or "").strip()
        assert text, f"chunk {row['id']} has no text; the dump would silently hide it"
        assert text in rendered


def test_main_writes_both_files_and_reports(corpus, tmp_path):
    root, _ = corpus

    assert dump.main(["--root", str(root), "--out-dir", str(tmp_path)]) == 0

    chunks = tmp_path / "chunks.txt"
    vectors = tmp_path / "embeddings.txt"
    assert chunks.exists() and vectors.exists()
    assert chunks.stat().st_size > 0
    assert vectors.stat().st_size > 0
    assert "20" in chunks.read_text(encoding="utf-8").splitlines()[5]


def test_what_chunks_skips_the_vector_file(corpus, tmp_path):
    root, _ = corpus

    assert dump.main(["--root", str(root), "--out-dir", str(tmp_path), "--what", "chunks"]) == 0
    assert (tmp_path / "chunks.txt").exists()
    assert not (tmp_path / "embeddings.txt").exists()


def test_no_vectors_flag_skips_the_vector_file(corpus, tmp_path):
    root, _ = corpus

    assert dump.main(["--root", str(root), "--out-dir", str(tmp_path), "--no-vectors"]) == 0
    assert (tmp_path / "chunks.txt").exists()
    assert not (tmp_path / "embeddings.txt").exists()
