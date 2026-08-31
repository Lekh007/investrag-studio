"""Pure filter-derivation and allowlist tests (design §11 steps 2-3)."""

from __future__ import annotations

import pytest

from investrag.filters import InvalidFilterError, derive_filters_and_rewrite, validate_filters


def test_allowlisted_filter_passes_through() -> None:
    assert validate_filters({"page": 3}) == {"page": 3}


def test_key_outside_the_allowlist_is_rejected() -> None:
    with pytest.raises(InvalidFilterError):
        validate_filters({"__proto__": "x"})


def test_wrong_value_type_is_rejected() -> None:
    with pytest.raises(InvalidFilterError):
        validate_filters({"page": "not-a-number"})


def test_page_hint_is_extracted_and_removed_from_the_query() -> None:
    rewritten, filters = derive_filters_and_rewrite("What was revenue on page:3?")
    assert filters == {"page": 3}
    assert "page:3" not in rewritten
    assert "revenue" in rewritten


def test_sheet_hint_is_extracted() -> None:
    rewritten, filters = derive_filters_and_rewrite("sheet:Summary what is the total")
    assert filters == {"sheet": "Summary"}
    assert "sheet:Summary" not in rewritten


def test_quoted_phrase_is_never_rewritten_even_if_it_looks_like_a_hint() -> None:
    rewritten, filters = derive_filters_and_rewrite('What does "page:3" mean in this filing?')
    assert filters == {}, "a quoted hint-like string must not be extracted as a real filter"
    assert '"page:3"' in rewritten


def test_malformed_hint_value_is_left_in_the_query_text() -> None:
    rewritten, filters = derive_filters_and_rewrite("summary for page:notanumber please")
    assert filters == {}
    assert "page:notanumber" in rewritten


def test_unrecognized_hint_key_is_left_in_the_query_text() -> None:
    rewritten, filters = derive_filters_and_rewrite("author:someone wrote this")
    assert filters == {}
    assert "author:someone" in rewritten


def test_question_with_no_hints_is_unchanged() -> None:
    rewritten, filters = derive_filters_and_rewrite("What was revenue for the quarter?")
    assert rewritten == "What was revenue for the quarter?"
    assert filters == {}
