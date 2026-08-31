from investrag.chunking import chunk_elements
from investrag.domain import CanonicalElement


def test_structure_aware_chunks_preserve_element_ids() -> None:
    elements = [
        CanonicalElement(element_id="h1", element_type="heading", text="Risk"),
        CanonicalElement(element_id="p1", element_type="paragraph", text="Credit spreads widened."),
    ]
    chunks = chunk_elements(elements, "src_1")
    assert len(chunks) == 1
    assert set(chunks[0].element_ids) == {"h1", "p1"}
    assert chunks[0].parent_id == "h1"


def _make_table_elements() -> list:
    return [
        CanonicalElement(element_id="h1", element_type="heading", text="Loan Portfolio"),
        CanonicalElement(element_id="p1", element_type="paragraph", text="Balances as of quarter end."),
        CanonicalElement(element_id="t1", element_type="table", text="table 4x3: Loan | Balance | Status"),
        CanonicalElement(element_id="r0", element_type="table_row", text="Loan | Balance | Status", cell_range="row:0"),
        CanonicalElement(element_id="r1", element_type="table_row", text="Loan 12 | $4.2M | Current", cell_range="row:1"),
        CanonicalElement(element_id="r2", element_type="table_row", text="Loan 30 | $2.1M | Watchlist", cell_range="row:2"),
        CanonicalElement(element_id="r3", element_type="table_row", text="Loan 47 | $6.8M | Current", cell_range="row:3"),
        CanonicalElement(element_id="r4", element_type="table_row", text="Loan 55 | $1.0M | Current", cell_range="row:4"),
        CanonicalElement(element_id="r5", element_type="table_row", text="Loan 61 | $3.3M | Watchlist", cell_range="row:5"),
        CanonicalElement(element_id="r6", element_type="table_row", text="Loan 72 | $0.9M | Current", cell_range="row:6"),
    ]


def test_parent_child_child_chunks_resolve_to_a_real_parent_chunk() -> None:
    elements = [
        CanonicalElement(element_id="h1", element_type="heading", text="Risk Factors"),
        CanonicalElement(element_id="p1", element_type="paragraph", text="Credit spreads widened materially this quarter across the portfolio."),
        CanonicalElement(element_id="p2", element_type="paragraph", text="Watchlist loans increased from four to seven during the period."),
    ]
    chunks = chunk_elements(elements, "src_pc", profile="parent-child")

    parents = [c for c in chunks if c.metadata.get("role") == "parent"]
    children = [c for c in chunks if c.metadata.get("role") == "child"]
    assert len(parents) == 1
    assert children, "a section must produce at least one child chunk"

    parent = parents[0]
    assert parent.parent_id is None
    assert "Credit spreads" in parent.text and "Watchlist loans" in parent.text, "parent must hold the FULL section, not a fragment"

    for child in children:
        assert child.parent_id == parent.chunk_id, "child.parent_id must resolve to a real, fetchable parent chunk id"
        assert len(child.text) <= 440  # _CHILD_MAX_CHARS plus overlap slack


def test_parent_child_produces_one_parent_per_heading_section() -> None:
    elements = [
        CanonicalElement(element_id="h1", element_type="heading", text="Section A"),
        CanonicalElement(element_id="p1", element_type="paragraph", text="Content of section A."),
        CanonicalElement(element_id="h2", element_type="heading", text="Section B"),
        CanonicalElement(element_id="p2", element_type="paragraph", text="Content of section B."),
    ]
    chunks = chunk_elements(elements, "src_pc2", profile="parent-child")
    parents = [c for c in chunks if c.metadata.get("role") == "parent"]
    assert len(parents) == 2
    assert {p.chunk_id for p in parents} == {c.parent_id for c in chunks if c.metadata.get("role") == "child"}


def test_table_aware_produces_a_summary_and_row_group_chunks() -> None:
    chunks = chunk_elements(_make_table_elements(), "src_ta", profile="table-aware")

    summaries = [c for c in chunks if c.metadata.get("role") == "table-summary"]
    row_groups = [c for c in chunks if c.metadata.get("role") == "table-row-group"]
    prose = [c for c in chunks if c.metadata.get("role") == "prose"]

    assert len(summaries) == 1
    # "nearby" narrative means the nearest preceding text, which is the paragraph
    # immediately before the table - not the heading two elements further back.
    assert "Balances as of quarter end" in summaries[0].text
    assert len(row_groups) == 2, "7 data rows incl. header at 5-per-group must split into 2 groups"
    assert prose, "non-table elements must still be chunked"

    second_group = row_groups[1]
    assert "Loan | Balance | Status" in second_group.text, "the header row must be prepended to groups that do not already contain it"
    assert second_group.metadata["cell_range"] and ".." in second_group.metadata["cell_range"]


def test_table_aware_falls_back_to_the_heading_when_no_paragraph_precedes_the_table() -> None:
    elements = [
        CanonicalElement(element_id="h1", element_type="heading", text="Loan Portfolio"),
        CanonicalElement(element_id="t1", element_type="table", text="table 2x2: Loan | Balance"),
        CanonicalElement(element_id="r0", element_type="table_row", text="Loan | Balance", cell_range="row:0"),
        CanonicalElement(element_id="r1", element_type="table_row", text="Loan 12 | $4.2M", cell_range="row:1"),
    ]
    chunks = chunk_elements(elements, "src_ta2", profile="table-aware")
    summary = next(c for c in chunks if c.metadata.get("role") == "table-summary")
    assert "Loan Portfolio" in summary.text


def test_chunk_profiles_registry_lists_all_four() -> None:
    from investrag.chunking import CHUNK_PROFILES

    assert set(CHUNK_PROFILES) == {"structure-aware", "recursive-baseline", "parent-child", "table-aware"}
    for entry in CHUNK_PROFILES.values():
        assert entry["version"] and entry["description"]
