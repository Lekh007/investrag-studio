from __future__ import annotations

import re

# Reuses the service-layer stopword philosophy: link/temporal words that would
# otherwise create spurious "topical overlap" between any two financial sentences.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "close", "did", "do", "does",
    "early", "estimates", "final", "for", "how", "in", "is", "it", "its", "many",
    "no", "not", "now", "of", "on", "or", "over", "placed", "quarter", "that",
    "the", "there", "third", "this", "to", "up", "was", "were", "what", "which",
    "who", "with", "year",
}
_SIGNIFICANT_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_NUMBER_UNIT_PATTERN = re.compile(r"\$?([\d][\d,]*(?:\.\d+)?)\s*(million|billion|percent|%)", re.IGNORECASE)


def _significant_words(text: str) -> set[str]:
    return {
        token.rstrip(".")
        for token in _SIGNIFICANT_WORD_PATTERN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _numbers_with_unit(text: str) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for match in _NUMBER_UNIT_PATTERN.finditer(text):
        value = match.group(1).replace(",", "")
        unit = match.group(2).lower().rstrip("%") or "%"
        normalized.append((value, unit))
    return normalized


def detect_conflicting_evidence(contexts: list[dict[str, str]]) -> tuple[bool, str | None]:
    """Real, bounded conflict signal for `QueryResponse.conflicting_evidence`
    (previously always False - never computed). Flags two retrieved contexts as
    conflicting when they share genuine topical vocabulary (not just link/temporal
    words) *and* report a different numeric value with the same unit - e.g. one
    source says revenue was "$412 million" and another says "$430 million".

    This is topic-level numeric divergence, not claim-level entailment: it cannot
    tell a genuine contradiction from two numbers about related-but-distinct facts
    that happen to share vocabulary. It is a real, testable signal that replaces an
    always-False stub, not a claim of semantic conflict resolution.
    """

    parsed = [(context["label"], _significant_words(context["text"]), _numbers_with_unit(context["text"])) for context in contexts]
    for i in range(len(parsed)):
        label_a, words_a, numbers_a = parsed[i]
        for j in range(i + 1, len(parsed)):
            label_b, words_b, numbers_b = parsed[j]
            if not (words_a & words_b):
                continue
            for value_a, unit_a in numbers_a:
                for value_b, unit_b in numbers_b:
                    if unit_a == unit_b and value_a != value_b:
                        return True, f"{label_a} reports {value_a}{unit_a} while {label_b} reports {value_b}{unit_b} for related content ({', '.join(sorted(words_a & words_b)[:3])})"
    return False, None
