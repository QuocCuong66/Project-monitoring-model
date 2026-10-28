"""User account management with bcrypt password hashing.

User credentials are stored locally in %APPDATA%/FallStrokeWarning/users.json.
The file is never committed to git and survives app reinstalls.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import bcrypt

from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
ROLE_ADMIN = "admin"
ROLE_VIEWER = "viewer"

VALID_ROLES = {ROLE_ADMIN, ROLE_VIEWER}


@dataclass
class UserAccount:
    """A local user account."""
    username: str
    password_hash: str
    role: str = ROLE_ADMIN
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    # Embedded in every token; bumping it (password change, logout) revokes all existing tokens
    token_version: int = 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _users_file_path() -> Path:
    """Path to the local users database file (%APPDATA%/FallStrokeWarning/users.json)."""
    try:
        folder = (load_config("notifications.yaml") or {}).get("user_settings_dir", "FallStrokeWarning")
    except Exception:
        folder = "FallStrokeWarning"
    base = Path(os.getenv("APPDATA") or Path.home())
    return base / folder / "users.json"


def _load_users() -> dict[str, dict[str, Any]]:
    path = _users_file_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict[str, dict[str, Any]]) -> None:
    path = _users_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_user(username: str, password: str, role: str = ROLE_ADMIN) -> UserAccount:
    """Create a new user account and persist to disk.

    Raises:
        ValueError: If the username already exists or role is invalid.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Vai trò không hợp lệ: {role}. Chọn: {VALID_ROLES}")

    users = _load_users()
    key = username.strip().lower()
    if key in users:
        raise ValueError(f"Tên đăng nhập '{username}' đã tồn tại.")

    account = UserAccount(
        username=key,
        password_hash=hash_password(password),
        role=role,
    )
    users[key] = asdict(account)
    _save_users(users)
    return account


def authenticate(username: str, password: str) -> Optional[UserAccount]:
    """Validate credentials and return the user account, or None if invalid."""
    users = _load_users()
    key = username.strip().lower()
    data = users.get(key)
    if data is None:
        return None

    if not verify_password(password, data["password_hash"]):
        return None

    # Update last_login timestamp
    data["last_login"] = time.time()
    users[key] = data
    _save_users(users)

    return UserAccount(**data)


def has_any_users() -> bool:
    """Check if at least one user account exists (first-run detection)."""
    return len(_load_users()) > 0


def get_user(username: str) -> Optional[UserAccount]:
    """Retrieve a user account by username."""
    users = _load_users()
    data = users.get(username.strip().lower())
    if data is None:
        return None
    return UserAccount(**data)


def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Change a user's password after verifying the old one."""
    users = _load_users()
    key = username.strip().lower()
    data = users.get(key)
    if data is None:
        return False
    if not verify_password(old_password, data["password_hash"]):
        return False
    data["password_hash"] = hash_password(new_password)
    data["token_version"] = int(data.get("token_version", 0)) + 1  # sign out every device
    users[key] = data
    _save_users(users)
    return True


def revoke_tokens(username: str) -> bool:
    """Invalidate every access / refresh token issued to this user so far."""
    users = _load_users()
    key = username.strip().lower()
    data = users.get(key)
    if data is None:
        return False
    data["token_version"] = int(data.get("token_version", 0)) + 1
    users[key] = data
    _save_users(users)
    return True


def is_token_current(payload: dict[str, Any]) -> bool:
    """A signed token is only valid while its user exists and has not revoked it."""
    data = _load_users().get(str(payload.get("sub", "")).strip().lower())
    if data is None:
        return False
    return int(payload.get("ver", 0)) == int(data.get("token_version", 0))


def list_users() -> list[dict[str, Any]]:
    """List all users (without password hashes)."""
    users = _load_users()
    result = []
    for data in users.values():
        safe = {k: v for k, v in data.items() if k != "password_hash"}
        result.append(safe)
    return result


def delete_user(username: str) -> bool:
    """Delete a user account."""
    users = _load_users()
    key = username.strip().lower()
    if key not in users:
        return False
    del users[key]
    _save_users(users)
    return True
