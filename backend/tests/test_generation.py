"""Generation-provenance tests.

The serving path stitches retrieved excerpts into a fallback answer that still
carries ``[S1]`` labels, so it passes the citation validator exactly like a real
completion. Without these tests an unreachable Ollama is indistinguishable from a
working model - which is what the suite previously asserted was correct.
"""

from pathlib import Path
from typing import Any

import httpx
import pytest

from investrag.config import Settings
from investrag.domain import QueryRequest
from investrag.llm import EXTRACTIVE_FALLBACK_PREAMBLE, LocalAnswerer
from investrag.service import InvestRAGService

CONTEXTS = [{"label": "S1", "text": "Revenue for the quarter was 412 million dollars."}]


def test_unreachable_model_is_reported_as_extractive_fallback() -> None:
    result = LocalAnswerer("http://127.0.0.1:1", "llama3.1:8b").answer("revenue?", CONTEXTS)
    assert result.mode == "extractive-fallback"
    assert result.used_fallback is True
    assert result.model is None, "a model that never ran must not be named as the author"
    assert result.fallback_reason
    assert result.text.startswith(EXTRACTIVE_FALLBACK_PREAMBLE)


def test_empty_completion_is_not_reported_as_a_model_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"response": "   "}, request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(httpx, "post", fake_post)
    result = LocalAnswerer("http://127.0.0.1:11434", "llama3.1:8b").answer("revenue?", CONTEXTS)
    assert result.mode == "extractive-fallback"
    assert result.fallback_reason == "model returned an empty completion"


def test_successful_generation_is_attributed_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"response": "Revenue was 412 million dollars [S1]."},
            request=httpx.Request("POST", "http://x"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = LocalAnswerer("http://127.0.0.1:11434", "llama3.1:8b").answer("revenue?", CONTEXTS)
    assert result.mode == "model"
    assert result.used_fallback is False
    assert result.model == "llama3.1:8b"
    assert result.fallback_reason is None


def test_generation_options_bound_repetition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for a live-measured bug (2026-08-30): granite4.2:3b, even with
    think: false, fell into an unbounded repetition loop on a multi-citation
    evidence prompt and reliably hit the 90s httpx timeout - reproduced 3/3
    through the real API. A raw /api/generate call with repeat_penalty set
    returned a correct answer in 8.8s instead. Reverting this options block
    reproduces the missing bound; asserting it here catches a regression without
    needing a live, slow model to fail against."""

    captured: dict[str, Any] = {}

    def fake_post(*_args: Any, **kwargs: Any) -> httpx.Response:
        captured.update(kwargs.get("json", {}))
        return httpx.Response(
            200, json={"response": "Revenue was 412 million dollars [S1]."}, request=httpx.Request("POST", "http://x")
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    LocalAnswerer("http://127.0.0.1:11434", "granite4.2:3b").answer("revenue?", CONTEXTS)
    options = captured["options"]
    assert options["repeat_penalty"] > 1.0, "no repeat_penalty lets the model loop until the timeout"
    assert isinstance(options["num_predict"], int) and options["num_predict"] > 0, (
        "no num_predict bound leaves a repetition loop unbounded even with a repeat_penalty"
    )


def _service_with_corpus(tmp_path: Path, base_url: str) -> InvestRAGService:
    service = InvestRAGService(
        Settings(
            data_dir=tmp_path,
            embedding_mode="hash",
            ollama_base_url=base_url,
            publish_vector_stores="",
        )
    )
    memo = tmp_path / "memo.md"
    memo.write_text("# Q3 memo\n\nRevenue for the quarter was 412 million dollars.\n", encoding="utf-8")
    source = service.ingest(memo)
    assert source.queryable, source.warnings
    return service


def test_degraded_generation_is_visible_in_the_query_response(tmp_path: Path) -> None:
    """The regression that mattered: a dead Ollama used to return a clean answer."""

    service = _service_with_corpus(tmp_path, "http://127.0.0.1:1")
    response = service.query(QueryRequest(question="What was revenue?", profile="hybrid", top_k=3))

    assert response.citations, "evidence retrieval still works while generation is down"
    assert response.trace.generation_mode == "extractive-fallback"
    assert response.trace.generation_fallback_reason
    assert any("not a generated answer" in message for message in response.validator_messages)


def test_working_generation_is_reported_as_a_model_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"response": "Revenue was 412 million dollars [S1]."},
            request=httpx.Request("POST", "http://x"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    service = _service_with_corpus(tmp_path, "http://127.0.0.1:11434")
    response = service.query(QueryRequest(question="What was revenue?", profile="hybrid", top_k=3))

    assert response.trace.generation_mode == "model"
    assert response.trace.generation_fallback_reason is None
    assert not any("not a generated answer" in message for message in response.validator_messages)


def _sequenced_post(replies: list[str]) -> Any:
    """Return an httpx.post stand-in that yields each reply in turn."""

    calls = {"n": 0}

    def fake_post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        index = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return httpx.Response(
            200, json={"response": replies[index]}, request=httpx.Request("POST", "http://x")
        )

    fake_post.calls = calls  # type: ignore[attr-defined]
    return fake_post


def test_uncited_first_draft_is_repaired_rather_than_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_post = _sequenced_post(
        ["Revenue was 412 million dollars.", "Revenue was 412 million dollars [S1]."]
    )
    monkeypatch.setattr(httpx, "post", fake_post)
    service = _service_with_corpus(tmp_path, "http://127.0.0.1:11434")
    response = service.query(QueryRequest(question="What was revenue?", profile="hybrid", top_k=3))

    assert fake_post.calls["n"] == 2, "the repair retry must actually be issued"
    assert response.trace.citation_repair_attempted is True
    assert response.trace.citation_repair_succeeded is True
    assert response.answer_withheld is False
    assert "[S1]" in response.answer


def test_answer_is_withheld_only_after_the_repair_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx, "post", _sequenced_post(["Revenue was 412 million dollars."]))
    service = _service_with_corpus(tmp_path, "http://127.0.0.1:11434")
    response = service.query(QueryRequest(question="What was revenue?", profile="hybrid", top_k=3))

    assert response.trace.citation_repair_attempted is True
    assert response.trace.citation_repair_succeeded is False
    assert response.answer_withheld is True
    assert any("repair retry was attempted" in m for m in response.validator_messages)
    assert not any("No indexed evidence matched" in m for m in response.validator_messages), (
        "evidence WAS retrieved; only the citation formatting failed"
    )


def test_invented_citation_label_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx, "post", _sequenced_post(["Revenue was 412 million dollars [S7]."]))
    service = _service_with_corpus(tmp_path, "http://127.0.0.1:11434")
    response = service.query(QueryRequest(question="What was revenue?", profile="hybrid", top_k=3))

    assert response.answer_withheld is True
    assert any("unknown citation labels" in m for m in response.validator_messages)


def test_model_is_never_called_when_retrieval_found_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked something off-corpus, llama3.1 invents a figure AND an [S1] label.

    Abstention must be decided by retrieval, so the model is not consulted at all.
    """

    fake_post = _sequenced_post(["The price of gold is $1,794.47 per ounce [S1]."])
    monkeypatch.setattr(httpx, "post", fake_post)
    service = _service_with_corpus(tmp_path, "http://127.0.0.1:11434")
    response = service.query(
        QueryRequest(question="What is the price of gold?", profile="hybrid", top_k=3)
    )

    assert fake_post.calls["n"] == 0, "no evidence must mean no generation call"
    assert response.insufficient_evidence is True
    assert response.citations == []
    assert response.trace.generation_mode == "abstained"
    assert "1,794.47" not in response.answer


def test_generate_query_variants_parses_model_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx, "post", _sequenced_post(["What was the operating margin?\nHow did margin change this quarter?"])
    )
    result = LocalAnswerer("http://127.0.0.1:11434", "granite4.2:3b").generate_query_variants("margin?")
    assert result.mode == "model"
    assert result.variants == ["What was the operating margin?", "How did margin change this quarter?"]
    assert result.fallback_reason is None


def test_generate_query_variants_falls_back_when_unreachable() -> None:
    result = LocalAnswerer("http://127.0.0.1:1", "granite4.2:3b").generate_query_variants("margin?")
    assert result.mode == "extractive-fallback"
    assert result.variants == []
    assert result.fallback_reason


def test_generate_query_variants_falls_back_on_empty_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", _sequenced_post(["   \n\n  "]))
    result = LocalAnswerer("http://127.0.0.1:11434", "granite4.2:3b").generate_query_variants("margin?")
    assert result.mode == "extractive-fallback"
    assert "no usable variant lines" in (result.fallback_reason or "")
