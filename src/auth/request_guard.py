"""Request origin checks shared by the HTTP middleware, auth routes and the alert WebSocket.

- Local access: only a loopback client that did NOT come through a reverse proxy / tunnel
  (ngrok, cloudflared, Tailscale Funnel... forward the real client in X-Forwarded-For etc.)
  counts as "this machine"; everything else must authenticate.
- Host check (anti DNS-rebinding): the Host header must name this machine (localhost, its
  hostname or one of its IP addresses) or an extra name from security.yaml auth.allowed_hosts
  (DDNS name, tunnel hostname). A malicious site rebinding its own domain to 127.0.0.1 is refused.
- Cross-site write check (anti CSRF): a browser request that changes state (POST / PUT / PATCH /
  DELETE) or opens the WebSocket must come from the dashboard's own origin, so a web page open
  in the user's browser cannot stop the camera or shut the system down through localhost.
"""

import os
import socket
import time
from typing import Any, Optional
from urllib.parse import urlsplit

from starlette.requests import HTTPConnection

from src.utils.config import load_config

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "cf-connecting-ip", "true-client-ip")
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

_LOCAL_NAMES_TTL_S = 60.0
_local_names_cache: tuple[float, set[str]] = (0.0, set())


def _auth_config() -> dict[str, Any]:
    try:
        return (load_config("security.yaml") or {}).get("auth", {}) or {}
    except Exception:
        return {}


def _cors_origins() -> list[str]:
    try:
        return (load_config("security.yaml") or {}).get("cors", {}).get("allowed_origins", []) or []
    except Exception:
        return []


def strip_port(host: str) -> str:
    """'example.com:8000' -> 'example.com', '[::1]:8000' -> '::1'."""
    host = (host or "").strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def _local_names() -> set[str]:
    """This machine's hostname and IP addresses (refreshed every minute: Wi-Fi / VPN IPs change)."""
    global _local_names_cache
    stamp, names = _local_names_cache
    if names and time.time() - stamp < _LOCAL_NAMES_TTL_S:
        return names
    names = set(LOOPBACK_HOSTS)
    try:
        hostname = socket.gethostname()
        names.update({hostname.lower(), socket.getfqdn().lower()})
        for info in socket.getaddrinfo(hostname, None):
            names.add(str(info[4][0]).split("%", 1)[0].lower())
    except OSError:
        pass
    _local_names_cache = (time.time(), names)
    return names


def allowed_hosts() -> set[str]:
    configured = list(_auth_config().get("allowed_hosts") or [])
    configured += os.getenv("ALLOWED_HOSTS", "").split(",")  # e.g. the cloud / DDNS domain
    extra = {strip_port(str(h)) for h in configured if str(h).strip()}
    return _local_names() | extra


def host_check_enabled() -> bool:
    return bool(_auth_config().get("host_check", True))


def is_allowed_host(conn: HTTPConnection) -> bool:
    if not host_check_enabled():
        return True
    return strip_port(conn.headers.get("host", "")) in allowed_hosts()


def is_local_request(conn: HTTPConnection) -> bool:
    """Loopback client that did not arrive through a proxy or tunnel."""
    if conn.client is None or conn.client.host.strip() not in LOOPBACK_HOSTS:
        return False
    return not any(conn.headers.get(h) for h in PROXY_HEADERS)


def _origin_allowed(origin: str, conn: HTTPConnection) -> bool:
    if origin in _cors_origins():
        return True
    parts = urlsplit(origin)
    if not parts.scheme or not parts.netloc:
        return False  # "null" (sandboxed iframe, file://) or garbage
    # Same origin as the request (dashboard served by this server, under any allowed host name)
    return parts.netloc.lower() == conn.headers.get("host", "").strip().lower() and is_allowed_host(conn)


def is_cross_site(conn: HTTPConnection) -> bool:
    """True when a browser sent this request from another site."""
    if conn.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return True
    origin: Optional[str] = conn.headers.get("origin")
    if origin is None:
        return False  # not a browser cross-origin request (curl, same-origin GET, native client)
    return not _origin_allowed(origin.strip(), conn)


def is_cross_site_write(conn: HTTPConnection, method: str) -> bool:
    return method.upper() in UNSAFE_METHODS and is_cross_site(conn)
