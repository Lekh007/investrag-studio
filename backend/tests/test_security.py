import zipfile
from pathlib import Path

import pytest

import investrag.security as security
from investrag.security import (
    UnsafeArchiveError,
    UnsafeURL,
    download_remote_file,
    safe_extract_zip,
    validate_remote_url,
)


def test_safe_zip_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, tmp_path / "out")


@pytest.mark.parametrize("url", ["ftp://example.com/file.txt", "http://127.0.0.1/file.txt", "http://localhost/file.txt", "http://user:secret@example.com/file.txt"])
def test_url_boundary_rejects_non_http_or_local_targets(url: str) -> None:
    with pytest.raises(UnsafeURL):
        validate_remote_url(url)


def test_remote_download_streams_and_removes_partial_oversize_file(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeResponse:
        status_code = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, _chunk_size: int):
            yield b"123"
            yield b"45"

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def stream(self, method: str, url: str) -> FakeResponse:
            calls.append((method, url))
            return FakeResponse()

    monkeypatch.setattr(security.httpx, "Client", FakeClient)
    monkeypatch.setattr(security, "validate_remote_url", lambda url: url)
    destination = tmp_path / "download.bin"
    with pytest.raises(UnsafeURL, match="size limit"):
        download_remote_file("https://example.com/download.bin", destination, max_bytes=4)
    assert calls == [("GET", "https://example.com/download.bin")]
    assert not destination.exists()
