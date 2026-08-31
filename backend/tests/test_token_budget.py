"""Real (unmocked) token-budget coverage. Uses granite4.2's actual tokenizer
(tokenizer files only, ~9s cold load, no model weights) when reachable; falls back
to a measured char-per-token estimate offline (design §20: core suite stays
network-free even though this specific test file needs the tokenizer download)."""

from __future__ import annotations

from investrag.token_budget import TokenBudget


def test_count_uses_the_real_tokenizer_when_available() -> None:
    budget = TokenBudget()
    if not budget.real_tokenizer_available:
        import pytest

        pytest.skip(f"tokenizer unavailable: {budget._unavailable_reason}")
    text = "Revenue for the quarter was $412 million, up 18 percent year over year."
    count = budget.count(text)
    assert 10 < count < 25, f"expected a realistic token count for this sentence, got {count}"


def test_fit_returns_text_unchanged_when_it_fits() -> None:
    budget = TokenBudget()
    text, truncated = budget.fit("short text", remaining_tokens=1000)
    assert text == "short text"
    assert truncated is False


def test_fit_truncates_and_reports_truncation_when_over_budget() -> None:
    budget = TokenBudget()
    long_text = "Revenue was 412 million dollars. " * 200
    text, truncated = budget.fit(long_text, remaining_tokens=20)
    assert truncated is True
    assert len(text) < len(long_text)
    assert text.endswith("…")


def test_fit_with_zero_remaining_tokens_returns_empty() -> None:
    budget = TokenBudget()
    text, truncated = budget.fit("anything", remaining_tokens=0)
    assert text == ""
    assert truncated is True


def test_fallback_estimate_still_works_when_tokenizer_is_unavailable() -> None:
    budget = TokenBudget(model_name="this-model-does-not-exist/definitely-not-real")
    assert budget.real_tokenizer_available is False
    count = budget.count("twelve characters here")
    assert count > 0
    text, truncated = budget.fit("a" * 1000, remaining_tokens=10)
    assert truncated is True
    assert len(text) < 1000
