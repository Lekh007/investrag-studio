from __future__ import annotations

import re

# Design §11 step 3: an explicit allowlist of filterable metadata keys and value
# types. Anything outside this allowlist is rejected, never passed through - an
# unvalidated filter derived from free text is exactly the kind of injection surface
# design §18 asks retrieval to guard against ("retrieved documents are untrusted
# evidence... prompt-like text inside documents cannot change system behavior" - the
# same discipline applies to a user's own query text deriving filters).
ALLOWED_FILTER_KEYS: dict[str, type] = {
    "source_id": str,
    "page": int,
    "sheet": str,
    "slide": int,
}


class InvalidFilterError(ValueError):
    pass


def validate_filters(filters: dict[str, object]) -> dict[str, object]:
    """Reject any filter key or value type outside the allowlist. Called on every
    filter dict before it reaches a retriever, whether derived from a query or
    supplied directly."""

    validated: dict[str, object] = {}
    for key, value in filters.items():
        expected_type = ALLOWED_FILTER_KEYS.get(key)
        if expected_type is None:
            raise InvalidFilterError(f"filter key '{key}' is not in the allowlist: {sorted(ALLOWED_FILTER_KEYS)}")
        if not isinstance(value, expected_type):
            raise InvalidFilterError(f"filter '{key}' expected {expected_type.__name__}, got {type(value).__name__}")
        validated[key] = value
    return validated


_FILTER_HINT_PATTERN = re.compile(r"\b(page|sheet|slide|source_id)\s*:\s*([\w.-]+)", re.IGNORECASE)
_QUOTED_PATTERN = re.compile(r'"[^"]*"')


def derive_filters_and_rewrite(question: str) -> tuple[str, dict[str, object]]:
    """Design §11 steps 2-3: derive metadata filters from explicit in-query hints
    (``page:3``, ``sheet:Summary``) and rewrite the question with those hints removed,
    so they do not pollute dense/lexical search as ordinary query tokens. Quoted
    phrases are located first and left untouched by the hint-removal pass - they are
    often the exact terms a user wants matched verbatim, never something to rewrite.

    Malformed hints (unrecognized key, wrong value type) are left in the query text
    unchanged rather than silently dropped or guessed at.
    """

    quoted_spans = [match.span() for match in _QUOTED_PATTERN.finditer(question)]

    def _in_quotes(position: int) -> bool:
        return any(start <= position < end for start, end in quoted_spans)

    filters: dict[str, object] = {}

    def _replace(match: re.Match[str]) -> str:
        if _in_quotes(match.start()):
            return match.group(0)
        key = match.group(1).lower()
        raw_value = match.group(2)
        expected_type = ALLOWED_FILTER_KEYS.get(key)
        if expected_type is None:
            return match.group(0)
        try:
            value: object = int(raw_value) if expected_type is int else raw_value
        except ValueError:
            return match.group(0)
        filters[key] = value
        return ""

    rewritten = _FILTER_HINT_PATTERN.sub(_replace, question)
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return (rewritten if rewritten else question), validate_filters(filters)
