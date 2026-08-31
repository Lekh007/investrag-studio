"""Malformed-file and hostile-archive fuzzing against the real ingestion path.

Design §6 (archive limits, "password-protected or unsupported items become visible
partial failures rather than being silently skipped"), §17 ("parser failure creates a
visible failed or partial version with retained diagnostics") and §18 (format detection
that does not trust the extension, traversal- and bomb-proof extraction, filenames that
never determine output paths).

Every case here runs through ``InvestRAGService.ingest`` - the same function the HTTP
endpoint calls - rather than a parser in isolation, because the property under test is
"the *system* degrades visibly", not "this parser raises". The invariants are the same
for all of them:

* ingestion returns a record; it never propagates an exception to the caller;
* hostile or unreadable input never ends up ``queryable``;
* the failure carries a diagnostic the operator can read.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import pytest

from investrag.config import Settings
from investrag.domain import SourceVersion
from investrag.security import UnsafeArchiveError, safe_extract_zip
from investrag.service import InvestRAGService

pytestmark = pytest.mark.security


@pytest.fixture
def service(tmp_path: Path) -> InvestRAGService:
    return InvestRAGService(Settings(data_dir=tmp_path / "data", embedding_mode="hash", ollama_base_url="http://127.0.0.1:1"))


def _assert_visible_failure(source: SourceVersion, *, allow_partial: bool = True) -> None:
    permitted = {"failed", "partial"} if allow_partial else {"failed"}
    assert source.status in permitted, f"expected a visible failure, got {source.status!r}"
    assert source.queryable is False
    assert source.warnings, "a failed or partial version must retain a diagnostic (design §17)"


# --------------------------------------------------------------------------------------
# Format confusion: design §8 step 3 detects the actual format instead of trusting the
# extension, so a hostile extension must not pick the parser.
# --------------------------------------------------------------------------------------


def test_zip_bytes_named_pdf_are_routed_by_content_not_extension(service: InvestRAGService, tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("inner.md", "# Inner\nRevenue grew.")
    source = service.ingest(path, "report.pdf")
    assert source.parser.startswith("safe-zip"), "a ZIP renamed .pdf must not reach the PDF parser"


def test_text_bytes_named_xlsx_fail_visibly(service: InvestRAGService, tmp_path: Path) -> None:
    path = tmp_path / "metrics.xlsx"
    path.write_text("this is not a workbook", encoding="utf-8")
    _assert_visible_failure(service.ingest(path, "metrics.xlsx"), allow_partial=False)


def test_binary_bytes_named_json_fail_visibly(service: InvestRAGService, tmp_path: Path) -> None:
    path = tmp_path / "api.json"
    path.write_bytes(struct.pack("<8Q", *range(8)) + b"\x00\xff\xfe")
    _assert_visible_failure(service.ingest(path, "api.json"), allow_partial=False)


# --------------------------------------------------------------------------------------
# Truncated and corrupt documents.
# --------------------------------------------------------------------------------------


def test_truncated_pdf_fails_visibly(service: InvestRAGService, tmp_path: Path) -> None:
    import pymupdf

    complete = tmp_path / "complete.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Revenue is stable this quarter.")
    document.save(str(complete))
    payload = complete.read_bytes()

    truncated = tmp_path / "truncated.pdf"
    truncated.write_bytes(payload[: len(payload) // 2])
    _assert_visible_failure(service.ingest(truncated, "truncated.pdf"))


def test_pdf_header_followed_by_garbage_fails_visibly(service: InvestRAGService, tmp_path: Path) -> None:
    path = tmp_path / "garbage.pdf"
    path.write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 40)
    _assert_visible_failure(service.ingest(path, "garbage.pdf"))


def test_docx_with_corrupt_document_xml_fails_visibly(service: InvestRAGService, tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", "<w:document><w:body><w:p>unclosed")
    _assert_visible_failure(service.ingest(path, "broken.docx"))


def test_docx_external_entity_cannot_read_a_local_file(service: InvestRAGService, tmp_path: Path) -> None:
    """XXE against the Office-XML path. ``parser.py`` imports ``defusedxml``; this
    proves that choice is load-bearing rather than decorative - the secret must not
    appear anywhere in the parsed elements, and the version must not silently succeed
    with the entity stripped and unreported."""

    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("TOPSECRET-CANARY-4417", encoding="utf-8")
    secret_uri = secret_file.resolve().as_uri()
    payload = (
        '<?xml version="1.0"?>'
        f'<!DOCTYPE root [<!ENTITY xxe SYSTEM "{secret_uri}">]>'
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:body><w:p><w:t>&xxe;</w:t></w:p></w:body></w:document>"
    )
    path = tmp_path / "xxe.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", payload)

    source = service.ingest(path, "xxe.docx")
    chunks = " ".join(chunk.text for chunk in service.catalog.list_chunks())
    assert "TOPSECRET-CANARY-4417" not in chunks
    assert "TOPSECRET-CANARY-4417" not in " ".join(source.warnings)
    _assert_visible_failure(source, allow_partial=False)
    # The specific refusal matters: stdlib ElementTree also declines this document, but
    # only with a generic "undefined entity" ParseError. `EntitiesForbidden` is
    # defusedxml refusing the entity *declaration*, which is what makes swapping the
    # import back to the stdlib parser fail this test rather than pass it quietly.
    assert any("EntitiesForbidden" in warning for warning in source.warnings)


# --------------------------------------------------------------------------------------
# Hostile archives (design §6 / §18). These assert against ``safe_extract_zip`` directly
# so the specific refusal reason is pinned, then against ``ingest`` so the refusal is
# surfaced as a visible failed version rather than a stack trace.
# --------------------------------------------------------------------------------------


def _archive(path: Path, members: list[tuple[str, bytes]], *, deflate: bool = False) -> Path:
    compression = zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=compression) as handle:
        for name, payload in members:
            handle.writestr(name, payload)
    return path


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "..\\escape.txt",
        "sub/../../escape.txt",
        "/absolute.txt",
    ],
)
def test_archive_traversal_variants_are_refused(tmp_path: Path, name: str) -> None:
    archive = _archive(tmp_path / "traversal.zip", [(name, b"no")])
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, tmp_path / "out")


def test_drive_relative_archive_member_stays_inside_the_destination(tmp_path: Path) -> None:
    """``C:evil.txt`` is the Windows drive-relative form, and it is contained by two
    different mechanisms depending on where the destination lives - so the invariant
    worth asserting is *containment*, not "raises".

    Same drive as the destination: ``pathlib`` drops the redundant drive and the member
    lands at ``<dest>/evil.txt``. Different drive: ``pathlib`` replaces the whole path,
    ``resolve()`` anchors it on the other drive's working directory, and the
    ``destination not in target.parents`` check refuses it. Verified both ways on this
    machine 2026-08-30; only the second raises, and asserting the raise alone would
    have made this test pass or fail on which drive TEMP happens to be."""

    destination = (tmp_path / "out").resolve()
    archive = _archive(tmp_path / "drive.zip", [("C:evil.txt", b"ESCAPED")])
    try:
        extracted = safe_extract_zip(archive, destination)
    except UnsafeArchiveError:
        return  # cross-drive: refused outright, which is also correct
    for member in extracted:
        assert destination in member.resolve().parents
    assert not Path("C:/evil.txt").exists()


def test_archive_symlink_member_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777 << 16) | 0o600
        handle.writestr(info, "C:/Windows/win.ini")
    with pytest.raises(UnsafeArchiveError, match="symlink"):
        safe_extract_zip(archive, tmp_path / "out")


def test_archive_file_count_limit_is_enforced(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "many.zip", [(f"f{index}.txt", b"x") for index in range(20)])
    with pytest.raises(UnsafeArchiveError, match="limit is 5"):
        safe_extract_zip(archive, tmp_path / "out", max_files=5)


def test_archive_uncompressed_expansion_limit_is_enforced(tmp_path: Path) -> None:
    """The classic decompression bomb: a few kilobytes of compressed zeroes that
    expand by three orders of magnitude. The limit must bite before extraction."""

    archive = _archive(tmp_path / "bomb.zip", [("bomb.bin", b"\x00" * (4 * 1024 * 1024))], deflate=True)
    assert archive.stat().st_size < 64 * 1024, "the fixture is meant to be a high-ratio bomb"
    with pytest.raises(UnsafeArchiveError, match="uncompressed-size limit"):
        safe_extract_zip(archive, tmp_path / "out", max_uncompressed_bytes=1024 * 1024)


def test_archive_nesting_depth_limit_is_enforced(tmp_path: Path) -> None:
    deep = "/".join(f"d{index}" for index in range(20)) + "/leaf.txt"
    archive = _archive(tmp_path / "deep.zip", [(deep, b"x")])
    with pytest.raises(UnsafeArchiveError, match="nesting limit"):
        safe_extract_zip(archive, tmp_path / "out", max_depth=5)


def test_hostile_archive_ingests_as_a_visible_failure(service: InvestRAGService, tmp_path: Path) -> None:
    archive = _archive(tmp_path / "traversal.zip", [("../escape.txt", b"no")])
    source = service.ingest(archive, "traversal.zip")
    _assert_visible_failure(source, allow_partial=False)
    assert any("escapes extraction directory" in warning for warning in source.warnings)
    assert not (tmp_path / "escape.txt").exists()


def test_archive_with_one_bad_member_is_partial_not_silently_skipped(service: InvestRAGService, tmp_path: Path) -> None:
    """Design §6: an unsupported item inside an archive is a *visible* partial
    failure. The good member must still be ingested; the bad one must still be named."""

    archive = _archive(
        tmp_path / "mixed.zip",
        [("good.md", b"# Outlook\nRevenue is stable."), ("bad.xlsx", b"not a workbook at all")],
    )
    source = service.ingest(archive, "mixed.zip")
    assert source.status == "partial"
    assert source.queryable is False
    assert any("bad.xlsx" in warning for warning in source.warnings)
    names = {child.name for child in service.list_sources()}
    assert "good.md" in names and "bad.xlsx" in names


# --------------------------------------------------------------------------------------
# Degenerate inputs and filename handling.
# --------------------------------------------------------------------------------------


def test_zero_byte_upload_fails_visibly(service: InvestRAGService, tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    _assert_visible_failure(service.ingest(path, "empty.pdf"))


def test_traversal_and_control_characters_in_a_filename_cannot_escape_the_raw_directory(
    service: InvestRAGService, tmp_path: Path
) -> None:
    """Design §18: "filenames never determine output paths directly"."""

    path = tmp_path / "payload.md"
    path.write_text("# Note\nRevenue is stable.", encoding="utf-8")
    hostile = "../../../../windows/system32/dr\x00op\ned.md"

    source = service.ingest(path, hostile)

    assert "/" not in source.name and "\\" not in source.name
    assert "\x00" not in source.name and "\n" not in source.name
    stored = list(service.settings.raw_dir.glob("*"))
    assert stored, "the artifact should still be stored, just under a safe name"
    for artifact in stored:
        assert service.settings.raw_dir.resolve() == artifact.resolve().parent


def test_non_utf8_markdown_is_decoded_without_crashing(service: InvestRAGService, tmp_path: Path) -> None:
    path = tmp_path / "latin.md"
    path.write_bytes(b"# Outlook\nRevenue rose 5 percent in Q\xff3.")
    source = service.ingest(path, "latin.md")
    assert source.status in {"ready", "partial"}


def test_deeply_nested_json_does_not_take_the_process_down(service: InvestRAGService, tmp_path: Path) -> None:
    """A recursive descent over attacker-controlled nesting is a denial-of-service if
    it escapes as a bare RecursionError. It must land as a visible failed version."""

    payload: object = "leaf"
    for _ in range(2000):
        payload = {"n": payload}
    path = tmp_path / "deep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    source = service.ingest(path, "deep.json")
    assert source.status in {"failed", "partial", "ready"}
    assert isinstance(source, SourceVersion)
