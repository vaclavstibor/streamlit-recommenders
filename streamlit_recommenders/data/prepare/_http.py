"""HTTPS opening with certifi's CA bundle for dataset downloads."""

from __future__ import annotations

import ssl
import urllib.request
from functools import lru_cache
from typing import Any

import certifi


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """Return a verifying SSL context trusting the system store plus certifi.

    Python on Windows only sees root certificates already present in the
    system store, which Windows populates lazily, so a valid chain (e.g.
    GroupLens's emSign root) can fail with CERTIFICATE_VERIFY_FAILED.
    Adding certifi's Mozilla bundle fixes that, while keeping the system
    store means roots installed by an institution (TLS-inspecting proxies)
    are still trusted.
    """
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def urlopen(url: str | urllib.request.Request, *, timeout: float = 60) -> Any:
    """Open ``url`` like ``urllib.request.urlopen``, verifying against system + certifi roots.

    Args:
        url: URL string or prepared ``Request``.
        timeout: Socket timeout in seconds.

    Returns:
        The HTTP response object (use as a context manager).
    """
    return urllib.request.urlopen(url, timeout=timeout, context=_ssl_context())
