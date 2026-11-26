"""Authentication REST API endpoints.

Endpoints:
    POST /api/auth/setup    — First-run: create the initial admin account (only from this machine)
    POST /api/auth/login    — Authenticate and receive JWT tokens
    POST /api/auth/refresh  — Exchange refresh token for new access token
    POST /api/auth/change-password — Change current user's password (signs out other devices)
    POST /api/auth/logout   — Revoke every token of the current user
    GET  /api/auth/status   — Check auth status (enabled, has users, current session)
    GET  /api/auth/me       — Get current logged-in user info
"""

import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from src.auth.users import (
    authenticate,
    change_password,
    create_user,
    get_user,
    has_any_users,
    is_token_current,
    revoke_tokens,
    ROLE_ADMIN,
)
from src.auth.jwt_handler import (
    access_token_max_age_seconds,
    create_access_token,
    create_refresh_token,
    verify_refresh_token,
)
from src.auth.middleware import extract_token, verify_session_token
from src.auth.request_guard import is_local_request
from src.utils.config import load_config

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

MIN_PASSWORD_LENGTH = 8


def _auth_enabled() -> bool:
    try:
        return bool((load_config("security.yaml") or {}).get("auth", {}).get("enabled", True))
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Brute-force protection: per-client sliding window (security.yaml rate_limit.login)
# ---------------------------------------------------------------------------

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def _parse_rate(rate: str) -> tuple[int, float]:
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(second|minute|hour|day)s?\s*", str(rate))
    if not match:
        return 5, 60.0
    return int(match.group(1)), float(_UNIT_SECONDS[match.group(2)])


class AttemptThrottle:
    """At most `limit` attempts per client per window; raises 429 beyond that."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: dict[str, deque] = defaultdict(deque)

    def _settings(self) -> tuple[bool, int, float]:
        try:
            cfg = (load_config("security.yaml") or {}).get("rate_limit", {}) or {}
        except Exception:
            cfg = {}
        limit, window = _parse_rate(cfg.get("login", "5/minute"))
        return bool(cfg.get("enabled", True)), limit, window

    def check(self, request: Request) -> None:
        enabled, limit, window = self._settings()
        if not enabled:
            return
        client = request.client.host if request.client else "unknown"
        now = time.time()
        with self._lock:
            attempts = self._attempts[client]
            while attempts and now - attempts[0] > window:
                attempts.popleft()
            if len(attempts) >= limit:
                retry_after = int(window - (now - attempts[0])) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Thử quá nhiều lần. Vui lòng đợi {retry_after} giây rồi thử lại.",
                    headers={"Retry-After": str(retry_after)},
                )
            attempts.append(now)

    def reset(self) -> None:
        with self._lock:
            self._attempts.clear()


login_throttle = AttemptThrottle()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SetupRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_session_cookie(response: Response, access_token: str) -> None:
    # Cookie for the MJPEG <img> stream; lives exactly as long as the access token
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="strict",
        max_age=access_token_max_age_seconds(),
    )


def _issue_tokens(username: str, role: str, version: int, response: Response) -> dict[str, str]:
    access_token = create_access_token(username, role, version)
    refresh_token = create_refresh_token(username, version)
    _set_session_cookie(response, access_token)
    return {"access_token": access_token, "refresh_token": refresh_token}


def _check_new_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=400, detail=f"Mật khẩu phải có ít nhất {MIN_PASSWORD_LENGTH} ký tự.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
def auth_status(request: Request) -> dict[str, Any]:
    """Public: check whether auth is enabled, whether initial setup is needed,
    client connection origin (local vs LAN), and the current session (if any)."""
    enabled = _auth_enabled()
    has_users = has_any_users()

    cfg = load_config("security.yaml") or {}
    allow_local = bool(cfg.get("auth", {}).get("allow_localhost_unauthenticated", True))

    is_local = is_local_request(request)
    can_access_without_login = is_local and allow_local

    # Try to extract current session info
    current_user: Optional[dict[str, Any]] = None
    token = extract_token(request)
    payload = verify_session_token(token) if token else None
    if payload:
        current_user = {
            "username": payload.get("sub"),
            "role": payload.get("role"),
            "is_local": False,
        }
    elif can_access_without_login:
        current_user = {
            "username": "local_admin",
            "role": "admin",
            "is_local": True,
        }

    return {
        "auth_enabled": enabled,
        "is_localhost": is_local,
        "can_access_without_login": can_access_without_login,
        "setup_required": enabled and not has_users,
        # The first admin can only be created on the machine running the system
        "setup_allowed_here": enabled and not has_users and is_local,
        "has_users": has_users,
        "current_user": current_user,
    }


@router.post("/setup")
def initial_setup(req: SetupRequest, request: Request, response: Response) -> dict[str, Any]:
    """First-run only: create the initial admin account, from this machine only.

    Otherwise any device on the Wi-Fi could claim the admin account before the owner did
    (the owner never needs one on this machine thanks to the localhost bypass).
    This endpoint is locked after the first user is created.
    """
    if not _auth_enabled():
        raise HTTPException(status_code=400, detail="Xác thực đã bị tắt trong cấu hình.")
    login_throttle.check(request)

    if not is_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="Tài khoản quản trị đầu tiên chỉ được tạo trên chính máy tính đang chạy hệ thống.",
        )

    if has_any_users():
        raise HTTPException(
            status_code=403,
            detail="Tài khoản admin đã được tạo. Dùng trang đăng nhập.",
        )

    username = req.username.strip()
    password = req.password.strip()

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Tên đăng nhập phải có ít nhất 3 ký tự.")
    _check_new_password(password)

    try:
        user = create_user(username, password, role=ROLE_ADMIN)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "message": f"Tài khoản admin '{user.username}' đã được tạo thành công!",
        **_issue_tokens(user.username, user.role, user.token_version, response),
        "user": {"username": user.username, "role": user.role},
    }


@router.post("/login")
def login(req: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    """Authenticate user and return JWT tokens (access token short-lived, refresh token long-lived)."""
    if not _auth_enabled():
        raise HTTPException(status_code=400, detail="Xác thực đã bị tắt trong cấu hình.")
    login_throttle.check(request)

    user = authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Sai tên đăng nhập hoặc mật khẩu.",
        )

    return {
        "status": "success",
        "message": f"Đăng nhập thành công! Xin chào, {user.username}.",
        **_issue_tokens(user.username, user.role, user.token_version, response),
        "user": {"username": user.username, "role": user.role},
    }


@router.post("/refresh")
def refresh_tokens(req: RefreshRequest, request: Request, response: Response) -> dict[str, Any]:
    """Exchange a valid, non-revoked refresh token for a new access token."""
    login_throttle.check(request)
    payload = verify_refresh_token(req.refresh_token)
    if payload is None or not is_token_current(payload):
        raise HTTPException(status_code=401, detail="Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.")

    user = get_user(payload.get("sub", ""))
    if user is None:
        raise HTTPException(status_code=401, detail="Tài khoản không tồn tại.")

    new_access = create_access_token(user.username, user.role, user.token_version)
    _set_session_cookie(response, new_access)

    return {
        "status": "success",
        "access_token": new_access,
        "user": {"username": user.username, "role": user.role},
    }


@router.post("/change-password")
def change_pwd(req: ChangePasswordRequest, request: Request, response: Response) -> dict[str, Any]:
    """Change the current user's password. Every other device is signed out; this one gets new tokens."""
    user_info = getattr(request.state, "user", None)
    if user_info is None or user_info.get("is_local"):
        raise HTTPException(status_code=401, detail="Chưa đăng nhập bằng tài khoản.")
    login_throttle.check(request)

    username = user_info.get("sub", "")
    _check_new_password(req.new_password.strip())

    success = change_password(username, req.old_password, req.new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Mật khẩu cũ không đúng.")

    user = get_user(username)
    return {
        "status": "success",
        "message": "Đổi mật khẩu thành công! Các thiết bị khác đã bị đăng xuất.",
        **_issue_tokens(user.username, user.role, user.token_version, response),
    }


@router.get("/me")
def get_current_user(request: Request) -> dict[str, Any]:
    """Get current logged-in user info."""
    user_info = getattr(request.state, "user", None)
    if user_info is None:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập.")
    return {
        "username": user_info.get("sub"),
        "role": user_info.get("role"),
    }


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, Any]:
    """Revoke every token of the current user (all devices) and clear the auth cookie."""
    user_info = getattr(request.state, "user", None)
    if user_info is not None and not user_info.get("is_local"):
        revoke_tokens(user_info.get("sub", ""))
    response.delete_cookie("access_token")
    return {"status": "success", "message": "Đã đăng xuất."}
