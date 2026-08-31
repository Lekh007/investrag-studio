from __future__ import annotations

import ipaddress
import shutil
import socket
import zipfile
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

import httpx


class UnsafeArchiveError(ValueError):
    pass


class UnsafeURL(ValueError):
    pass


def validate_remote_url(url: str) -> str:
    """Validate an import URL against the localhost-only SSRF boundary."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURL("only absolute http(s) URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURL("URLs containing credentials are not allowed")
    hostname = parsed.hostname.rstrip(".").lower()
    blocked_names = {"localhost", "localhost.localdomain", "metadata.google.internal", "instance-data.ec2.internal"}
    if hostname in blocked_names or hostname.endswith(".local"):
        raise UnsafeURL("local and cloud-metadata hosts are not allowed")
    try:
        addresses = {cast(str, info[4][0]) for info in socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise UnsafeURL(f"hostname could not be resolved: {hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_unspecified:
            raise UnsafeURL("private, loopback, link-local and reserved destinations are not allowed")
    return url


def download_remote_file(url: str, destination: Path, *, max_bytes: int = 100 * 1024 * 1024, max_redirects: int = 3) -> str:
    """Download a bounded remote artifact while validating every redirect target."""

    current = validate_remote_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        for _ in range(max_redirects + 1):
            with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise UnsafeURL("redirect response did not include a location")
                    current = validate_remote_url(str(httpx.URL(current).join(location)))
                    continue
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            raise UnsafeURL("remote artifact exceeds configured size limit")
                    except ValueError:
                        pass
                total = 0
                try:
                    with destination.open("wb") as handle:
                        for chunk in response.iter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > max_bytes:
                                raise UnsafeURL("remote artifact exceeds configured size limit")
                            handle.write(chunk)
                except Exception:
                    destination.unlink(missing_ok=True)
                    raise
                filename = Path(urlparse(current).path).name or "remote-artifact.bin"
                return filename
    raise UnsafeURL("too many redirects")


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    max_files: int = 500,
    max_uncompressed_bytes: int = 250 * 1024 * 1024,
    max_depth: int = 12,
) -> list[Path]:
    """Extract a ZIP without path traversal, symlink or expansion-bomb surprises."""

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > max_files:
            raise UnsafeArchiveError(f"archive contains {len(members)} files; limit is {max_files}")
        for member in members:
            raw_name = member.filename.replace("\\", "/")
            path = Path(raw_name)
            if path.is_absolute() or ".." in path.parts:
                raise UnsafeArchiveError(f"archive member escapes extraction directory: {member.filename}")
            if len(path.parts) > max_depth:
                raise UnsafeArchiveError(f"archive member exceeds nesting limit: {member.filename}")
            is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
            if is_symlink:
                raise UnsafeArchiveError(f"symlink archive member is not allowed: {member.filename}")
            total_bytes += member.file_size
            if total_bytes > max_uncompressed_bytes:
                raise UnsafeArchiveError("archive exceeds uncompressed-size limit")
            target = (destination / path).resolve()
            if destination not in target.parents and target != destination:
                raise UnsafeArchiveError(f"archive member resolves outside destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)
    return extracted
