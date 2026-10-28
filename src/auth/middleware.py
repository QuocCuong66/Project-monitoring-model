"""FastAPI middleware that enforces JWT authentication on protected routes.

Every request first passes the request-origin checks (src.auth.request_guard): the Host header
must name this machine (anti DNS-rebinding) and browser writes must come from the dashboard's own
origin (anti CSRF). These run even with auth disabled, because the localhost bypass would
otherwise let any web page open in the user's browser drive the API.

Public routes (login, static assets, health check) are exempt from authentication.
The middleware extracts the Bearer token from the Authorization header, cookie or query string,
verifies it (signature, expiry, not revoked) and attaches the payload to request.state.user.
"""

from typing import Any, Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.auth.jwt_handler import verify_access_token
from src.auth.request_guard import is_allowed_host, is_cross_site_write, is_local_request
from src.auth.users import is_token_current
from src.utils.config import load_config


def _security_config() -> dict[str, Any]:
    try:
        return load_config("security.yaml") or {}
    except Exception:
        return {}


def _is_auth_enabled() -> bool:
    return bool(_security_config().get("auth", {}).get("enabled", True))


def _public_paths() -> list[str]:
    """Return the list of public (no-auth) path prefixes from config."""
    default = [
        "/health",
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/status",
        "/api/auth/refresh",
        "/",
        "/styles.css",
        "/app.js",
        "/favicon.ico",
        "/icon.png",
        "/logo.png",
        "/apple-touch-icon.png",
        "/static",
        "/huong-dan.html",
        "/guide",
    ]
    configured = _security_config().get("auth", {}).get("public_paths", default)
    return configured


def _is_public_route(path: str) -> bool:
    """Check if a request path matches a public route."""
    public = _public_paths()
    # Exact matches (e.g. "/", "/health", "/styles.css", "/api/auth/login")
    if path in public:
        return True
    # Prefix matches for directories (skip "/" so it doesn't match every path)
    for prefix in public:
        if prefix == "/":
            continue
        if prefix.endswith("/") or prefix in ("/static", "/docs"):
            if path.startswith(prefix):
                return True
    return False


def extract_token(request: Request) -> Optional[str]:
    """Extract Bearer token from Authorization header, cookie, or query parameter."""
    # 1. Authorization header (preferred)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    # 2. Cookie fallback (for MJPEG stream in <img> tag, which can't set headers)
    token_cookie = request.cookies.get("access_token")
    if token_cookie:
        return token_cookie
    # 3. Query param fallback (for <img> tags or streaming clients)
    token_query = request.query_params.get("token")
    if token_query:
        return token_query
    return None


def verify_session_token(token: str) -> Optional[dict[str, Any]]:
    """Valid signature + not expired + not revoked (password changed / logged out)."""
    payload = verify_access_token(token)
    if payload is None or not is_token_current(payload):
        return None
    return payload


def is_localhost_client(request: Request) -> bool:
    """This machine itself: loopback client that did not come through a proxy / tunnel."""
    return is_local_request(request)


def is_localhost_unauthenticated_allowed() -> bool:
    """Check if localhost clients are allowed to bypass authentication."""
    return bool(_security_config().get("auth", {}).get("allow_localhost_unauthenticated", True))


class AuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing request-origin checks and JWT Bearer authentication.

    Localhost access (127.0.0.1) can be configured to bypass authentication so the host
    machine runs with zero friction, while external/LAN connections strictly require valid JWT.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not is_allowed_host(request):
            return JSONResponse(
                status_code=400,
                content={"detail": "Tên máy chủ (Host) không hợp lệ. Thêm tên miền vào auth.allowed_hosts trong configs/security.yaml nếu cần."},
            )
        if is_cross_site_write(request, request.method):
            return JSONResponse(
                status_code=403,
                content={"detail": "Yêu cầu từ trang web khác bị từ chối (chống giả mạo yêu cầu)."},
            )

        # Skip auth entirely if disabled in config
        if not _is_auth_enabled():
            return await call_next(request)

        path = request.url.path

        # Allow public routes without auth
        if _is_public_route(path):
            return await call_next(request)

        # Allow WebSocket upgrades (handled separately in the WS endpoint)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        is_local = is_localhost_client(request)
        allow_local = is_localhost_unauthenticated_allowed()

        # Extract and verify token if present
        token = extract_token(request)
        if token is not None:
            payload = verify_session_token(token)
            if payload is not None:
                request.state.user = payload
                return await call_next(request)
            if not (is_local and allow_local):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại."},
                )

        # If accessing from localhost and localhost bypass is enabled: allow seamless local access
        if is_local and allow_local:
            request.state.user = {
                "sub": "local_admin",
                "role": "admin",
                "is_local": True,
            }
            return await call_next(request)

        # Remote LAN / Wi-Fi clients without token are rejected
        return JSONResponse(
            status_code=401,
            content={"detail": "Chưa đăng nhập. Truy cập từ mạng nội bộ (LAN/Wi-Fi) yêu cầu xác thực tài khoản."},
        )


def require_admin(request: Request) -> None:
    """Dependency: raise 403 if the current user is not an admin."""
    user = getattr(request.state, "user", None)
    if user is None or user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Chỉ tài khoản Quản trị viên (Admin) mới có quyền thực hiện thao tác này.",
        )
