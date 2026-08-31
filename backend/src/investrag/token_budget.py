from __future__ import annotations

from typing import Any

# Conservative fallback ratio when the real tokenizer cannot be loaded (offline, no
# network for the one-time HF download). Measured against granite4.2's own tokenizer
# on real evidence text, 2026-08-30: ~4.2 chars/token: this stays on the safe side
# (undercounts available room slightly rather than overcounts and risks truncating a
# citation label mid-token).
_CHARS_PER_TOKEN_ESTIMATE = 4.0


class TokenBudget:
    """Design §11 step 8: bound the evidence packet by the generation model's actual
    context window, not an approximate character count. Uses `granite4.2:3b`'s real
    tokenizer (`ibm-granite/granite-4.2-3b` on Hugging Face - tokenizer files only,
    ~9s to load, no model weights) when reachable, and a measured char-per-token
    fallback otherwise - offline test environments must still function (design §20).
    """

    def __init__(self, model_name: str = "ibm-granite/granite-4.2-3b", max_tokens: int = 8000) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._tokenizer: Any = None
        self._unavailable_reason: str | None = None

    def _load(self) -> Any:
        if self._tokenizer is not None or self._unavailable_reason is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as exc:
            self._unavailable_reason = f"{exc.__class__.__name__}: {exc}"
        return self._tokenizer

    @property
    def real_tokenizer_available(self) -> bool:
        self._load()
        return self._tokenizer is not None

    def count(self, text: str) -> int:
        tokenizer = self._load()
        if tokenizer is not None:
            return len(tokenizer.encode(text))
        return max(1, int(len(text) / _CHARS_PER_TOKEN_ESTIMATE))

    def fit(self, text: str, remaining_tokens: int) -> tuple[str, bool]:
        """Returns (text truncated to fit `remaining_tokens`, whether it was cut)."""

        if remaining_tokens <= 0:
            return "", True
        tokenizer = self._load()
        if tokenizer is not None:
            token_ids = tokenizer.encode(text)
            if len(token_ids) <= remaining_tokens:
                return text, False
            truncated = tokenizer.decode(token_ids[:remaining_tokens], skip_special_tokens=True)
            return truncated.rstrip() + " …", True
        char_budget = int(remaining_tokens * _CHARS_PER_TOKEN_ESTIMATE)
        if len(text) <= char_budget:
            return text, False
        return text[: max(1, char_budget - 2)].rstrip() + " …", True
