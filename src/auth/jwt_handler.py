"""JWT token creation and verification.

The secret key is automatically generated on first run and stored at
%APPDATA%/FallStrokeWarning/secret.key — never committed to git.
"""

import os
import secrets
import time
from pathlib import Path
from typing import Any, Optional

import jwt

from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Secret key management
# ---------------------------------------------------------------------------

def _secret_key_path() -> Path:
    """Path to the JWT secret key file."""
    try:
        folder = (load_config("notifications.yaml") or {}).get("user_settings_dir", "FallStrokeWarning")
    except Exception:
        folder = "FallStrokeWarning"
    base = Path(os.getenv("APPDATA") or Path.home())
    return base / folder / "secret.key"


def _load_or_create_secret() -> str:
    """Load existing secret key or generate a new one."""
    path = _secret_key_path()
    if path.exists():
        try:
            key = path.read_text(encoding="utf-8").strip()
            if len(key) >= 32:
                return key
        except OSError:
            pass
    # Generate a cryptographically secure random key
    key = secrets.token_hex(64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    return key


# Cache in module memory so we don't hit disk on every request
_SECRET_KEY: Optional[str] = None


def get_secret_key() -> str:
    """Get the JWT signing secret (lazy-loaded, cached)."""
    global _SECRET_KEY
    if _SECRET_KEY is None:
        _SECRET_KEY = _load_or_create_secret()
    return _SECRET_KEY


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _security_config() -> dict[str, Any]:
    try:
        return (load_config("security.yaml") or {}).get("auth", {}) or {}
    except Exception:
        return {}


def _algorithm() -> str:
    return _security_config().get("jwt_algorithm", "HS256")


def _access_expire_minutes() -> int:
    return int(_security_config().get("access_token_expire_minutes", 60))


def access_token_max_age_seconds() -> int:
    return _access_expire_minutes() * 60


def _refresh_expire_days() -> int:
    return int(_security_config().get("refresh_token_expire_days", 30))


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

def create_access_token(username: str, role: str, version: int = 0) -> str:
    """Create a short-lived access token (JWT). `version` = the user's token_version."""
    now = time.time()
    payload = {
        "sub": username,
        "role": role,
        "ver": int(version),
        "type": "access",
        "iat": now,
        "exp": now + _access_expire_minutes() * 60,
    }
    return jwt.encode(payload, get_secret_key(), algorithm=_algorithm())


def create_refresh_token(username: str, version: int = 0) -> str:
    """Create a long-lived refresh token (JWT). `version` = the user's token_version."""
    now = time.time()
    payload = {
        "sub": username,
        "ver": int(version),
        "type": "refresh",
        "iat": now,
        "exp": now + _refresh_expire_days() * 86400,
    }
    return jwt.encode(payload, get_secret_key(), algorithm=_algorithm())


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

def verify_access_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and verify an access token. Returns payload dict or None if invalid/expired."""
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[_algorithm()])
        if payload.get("type") != "access":
            return None
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def verify_refresh_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and verify a refresh token. Returns payload dict or None if invalid/expired."""
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[_algorithm()])
        if payload.get("type") != "refresh":
            return None
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
