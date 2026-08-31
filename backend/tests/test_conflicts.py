"""Real (unmocked) conflicting-evidence detection. Replaces the previous always-
False QueryResponse.conflicting_evidence stub."""

from __future__ import annotations

from investrag.conflicts import detect_conflicting_evidence


def test_detects_a_genuine_numeric_conflict_over_shared_topic() -> None:
    contexts = [
        {"label": "S1", "text": "Revenue for the third quarter was $412 million, up 18 percent year over year."},
        {"label": "S2", "text": "Early estimates placed third-quarter revenue at $430 million before final close."},
    ]
    conflicting, reason = detect_conflicting_evidence(contexts)
    assert conflicting is True
    assert "412" in reason and "430" in reason


def test_agreeing_sources_are_not_flagged() -> None:
    contexts = [
        {"label": "S1", "text": "Revenue grew 12 percent this quarter."},
        {"label": "S2", "text": "The report confirms revenue growth of 12 percent."},
    ]
    conflicting, reason = detect_conflicting_evidence(contexts)
    assert conflicting is False
    assert reason is None


def test_unrelated_numbers_are_not_flagged_as_conflicting() -> None:
    """Different topics with different numbers must not trigger a false positive -
    only genuine topical overlap (shared vocabulary) counts."""

    contexts = [
        {"label": "S1", "text": "Revenue for the quarter was $412 million."},
        {"label": "S2", "text": "The board approved a dividend of $1.25 per share."},
    ]
    conflicting, reason = detect_conflicting_evidence(contexts)
    assert conflicting is False


def test_single_context_cannot_conflict_with_itself() -> None:
    contexts = [{"label": "S1", "text": "Revenue was $412 million and margin was 22 percent."}]
    conflicting, reason = detect_conflicting_evidence(contexts)
    assert conflicting is False


def test_empty_contexts_do_not_conflict() -> None:
    assert detect_conflicting_evidence([]) == (False, None)


def test_conflicting_evidence_surfaces_end_to_end_through_a_real_query(tmp_path) -> None:
    """The same conflict scenario used in build_golden_set.py's fixture, run through
    the real service end to end - proves the signal reaches QueryResponse, not just
    the pure detector function."""

    from investrag.config import Settings
    from investrag.domain import QueryRequest
    from investrag.service import InvestRAGService

    settings = Settings(data_dir=tmp_path, embedding_mode="hash", ollama_base_url="http://127.0.0.1:1")
    service = InvestRAGService(settings)
    final = tmp_path / "final.md"
    final.write_text("# Q3 Final\nRevenue for the third quarter was $412 million.\n", encoding="utf-8")
    preliminary = tmp_path / "preliminary.md"
    preliminary.write_text("# Q3 Preliminary\nEarly estimates placed third-quarter revenue at $430 million.\n", encoding="utf-8")
    service.ingest(final)
    service.ingest(preliminary)

    response = service.query(QueryRequest(question="What was third quarter revenue?", profile="hybrid", top_k=5))
    assert response.conflicting_evidence is True, response.citations
    assert any("Conflicting evidence" in message for message in response.validator_messages)
