"""SSRF-safe HTTP fetching for the web loader.

* only ``http``/``https``, no credentials in the URL;
* the host must resolve **only** to public (globally routable) addresses - loopback, private,
  link-local (cloud metadata ``169.254.169.254``), CGNAT, multicast and reserved ranges are refused;
* every redirect hop is validated again, and the number of hops is limited;
* response size is capped while streaming; only textual content types are accepted.

Residual risk: DNS rebinding between the check and the actual connection is not prevented
(the check and ``requests`` resolve the name separately). Run the service without network access to
internal systems if that matters in your threat model.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

import requests

_ALLOWED_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", "text/xml", "application/xml")


class WebFetchError(Exception):
    """A URL could not be fetched. ``code`` is stable; ``message`` is safe to show to users."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def validate_public_url(url: str, *, allow_private: bool = False) -> str:
    """Return the normalised URL or raise :class:`WebFetchError`."""
    url = (url or "").strip()
    if not url or len(url) > 2048:
        raise WebFetchError("invalid_url", "Enter a valid http(s) URL (max 2048 characters).")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise WebFetchError("invalid_url", "The URL is malformed.") from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise WebFetchError("invalid_url", "Only http:// and https:// URLs are supported.")
    if parts.username or parts.password:
        raise WebFetchError("invalid_url", "URLs with embedded credentials are not allowed.")
    if allow_private:
        return url
    try:
        infos = socket.getaddrinfo(parts.hostname, port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebFetchError("network", f"The host '{parts.hostname}' could not be resolved.") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if not address.is_global:
            raise WebFetchError("blocked_host", "This address is not allowed: URLs that point to private or internal networks are blocked.")
    return url


class SafeSession(requests.Session):
    """``requests`` session that applies the rules above to the first request and to every redirect."""

    def __init__(self, *, max_bytes: int, max_redirects: int, allow_private: bool = False) -> None:
        super().__init__()
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.allow_private = allow_private
        self.trust_env = False  # ignore proxy environment variables

    def send(self, request, **kwargs):  # type: ignore[override]
        validate_public_url(request.url, allow_private=self.allow_private)
        kwargs["stream"] = True
        response = super().send(request, **kwargs)
        if response.is_redirect or response.is_permanent_redirect:
            return response  # the body is irrelevant; the next hop is validated by the next send()
        content_type = response.headers.get("content-type", "text/html").split(";")[0].strip().lower()
        if response.ok and not content_type.startswith(_ALLOWED_TYPES):
            response.close()
            raise WebFetchError("unsupported_content", f"Unsupported content type '{content_type}'. Only HTML and text pages can be imported.")
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            response.close()
            raise WebFetchError("too_large", f"The page is larger than the {self.max_bytes // (1024 * 1024)} MB limit.")
        body = bytearray()
        for chunk in response.iter_content(64 * 1024):
            body.extend(chunk)
            if len(body) > self.max_bytes:
                response.close()
                raise WebFetchError("too_large", f"The page is larger than the {self.max_bytes // (1024 * 1024)} MB limit.")
        response._content = bytes(body)
        response._content_consumed = True
        return response
