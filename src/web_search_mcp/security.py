from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata",
}
BLOCKED_SUFFIXES = (".local", ".internal", ".localhost", ".home", ".lan")
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}


class UnsafeUrlError(ValueError):
    """Raised when a URL could reach an internal or unsafe destination."""


def _is_public_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_url(url: str, *, resolve_dns: bool = True) -> str:
    """Validate an HTTP(S) URL and reject private/internal destinations."""

    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise UnsafeUrlError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URLs with embedded credentials are not allowed")
    if parsed.port not in ALLOWED_PORTS:
        raise UnsafeUrlError("Only ports 80 and 443 are allowed")

    host = parsed.hostname.lower().rstrip(".")
    if host in BLOCKED_HOSTS or host.endswith(BLOCKED_SUFFIXES):
        raise UnsafeUrlError(f"Blocked hostname: {host}")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and not _is_public_ip(str(ip)):
        raise UnsafeUrlError(f"Blocked IP address: {host}")

    if resolve_dns and ip is None:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(host, parsed.port or 443)
            }
        except socket.gaierror as exc:
            raise UnsafeUrlError(f"Could not resolve hostname: {host}") from exc
        if not addresses:
            raise UnsafeUrlError(f"Hostname resolved to no addresses: {host}")
        for address in addresses:
            if not _is_public_ip(address):
                raise UnsafeUrlError(
                    f"Hostname {host} resolves to a blocked address: {address}"
                )

    return url
