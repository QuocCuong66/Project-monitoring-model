"""Telegram Bot settings storage and API client for emergency notifications."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

from src.utils.config import get_env, load_config

_TOKEN_PATTERN = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")
_CHAT_ID_PATTERN = re.compile(r"^(-?\d+|@[A-Za-z0-9_]{5,})$")


class TelegramError(Exception):
    """Raised when the Telegram Bot API rejects a request or is unreachable."""


@dataclass
class TelegramSettings:
    """User-provided Telegram notification settings."""

    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""

    @property
    def is_ready(self) -> bool:
        return self.enabled and bool(self.bot_token) and bool(self.chat_id)

    def masked_token(self) -> str:
        if not self.bot_token:
            return ""
        bot_id, _, secret = self.bot_token.partition(":")
        return f"{bot_id}:{'*' * 6}{secret[-4:]}" if secret else "*" * 6


def is_valid_token(token: str) -> bool:
    return bool(_TOKEN_PATTERN.match(token.strip()))


def is_valid_chat_id(chat_id: str) -> bool:
    return bool(_CHAT_ID_PATTERN.match(chat_id.strip()))


def load_notification_config() -> dict[str, Any]:
    try:
        return load_config("notifications.yaml") or {}
    except Exception:
        return {}


def get_settings_path() -> Path:
    """Return the per-user file storing Telegram settings (outside the install dir)."""
    cfg = load_notification_config()
    base = Path(os.getenv("APPDATA") or Path.home())
    return base / cfg.get("user_settings_dir", "FallStrokeWarning") / "telegram.yaml"


def load_telegram_settings(path: Optional[Path] = None) -> TelegramSettings:
    """Load saved settings, falling back to TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars."""
    path = path or get_settings_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return TelegramSettings(
                enabled=bool(data.get("enabled", False)),
                bot_token=str(data.get("bot_token") or ""),
                chat_id=str(data.get("chat_id") or ""),
            )
        except Exception as e:
            print(f"[Warning] Cannot read Telegram settings at {path}: {e}")

    token = get_env("TELEGRAM_BOT_TOKEN") or ""
    chat_id = get_env("TELEGRAM_CHAT_ID") or ""
    return TelegramSettings(enabled=bool(token and chat_id), bot_token=token, chat_id=chat_id)


def save_telegram_settings(settings: TelegramSettings, path: Optional[Path] = None) -> Path:
    path = path or get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"enabled": settings.enabled, "bot_token": settings.bot_token, "chat_id": settings.chat_id},
            f,
            allow_unicode=True,
        )
    return path


class TelegramClient:
    """Minimal Telegram Bot API client (sendMessage, sendPhoto, getMe, getUpdates)."""

    def __init__(self, bot_token: str, config: Optional[dict[str, Any]] = None) -> None:
        cfg = (config if config is not None else load_notification_config()).get("telegram", {})
        self.bot_token = bot_token.strip()
        self.api_base_url = str(cfg.get("api_base_url", "https://api.telegram.org")).rstrip("/")
        self.timeout = float(cfg.get("request_timeout_seconds", 5.0))
        self.photo_timeout = float(cfg.get("photo_timeout_seconds", 10.0))

    def _call(self, method: str, timeout: Optional[float] = None, **kwargs: Any) -> Any:
        url = f"{self.api_base_url}/bot{self.bot_token}/{method}"
        try:
            resp = requests.post(url, timeout=timeout or self.timeout, **kwargs)
        except requests.RequestException as e:
            raise TelegramError("Không kết nối được tới máy chủ Telegram. Kiểm tra kết nối Internet.") from e
        try:
            body = resp.json()
        except ValueError as e:
            raise TelegramError(f"Telegram trả về phản hồi không hợp lệ (HTTP {resp.status_code}).") from e
        if not body.get("ok"):
            raise TelegramError(_describe_error(resp.status_code, str(body.get("description", ""))))
        return body.get("result")

    def get_bot_username(self) -> str:
        return str(self._call("getMe").get("username", ""))

    def find_chats(self) -> list[dict[str, str]]:
        """List chats that recently messaged the bot, newest first."""
        updates = self._call("getUpdates") or []
        chats: dict[str, dict[str, str]] = {}
        for update in reversed(updates):
            message = update.get("message") or update.get("my_chat_member") or update.get("channel_post") or {}
            chat = message.get("chat")
            if not chat:
                continue
            chat_id = str(chat.get("id"))
            if chat_id in chats:
                continue
            name = chat.get("title") or " ".join(
                part for part in (chat.get("first_name"), chat.get("last_name")) if part
            ) or chat.get("username") or chat_id
            chats[chat_id] = {"chat_id": chat_id, "name": name, "type": str(chat.get("type", ""))}
        return list(chats.values())

    def send_message(self, chat_id: str, text: str, parse_mode: Optional[str] = None) -> None:
        data = {"chat_id": chat_id, "text": text}
        if parse_mode:
            data["parse_mode"] = parse_mode
        self._call("sendMessage", data=data)

    def send_photo(self, chat_id: str, photo_path: Path, caption: str = "") -> None:
        with open(photo_path, "rb") as photo_file:
            self._call(
                "sendPhoto",
                timeout=self.photo_timeout,
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": photo_file},
            )


def _describe_error(status_code: int, description: str) -> str:
    if status_code == 401:
        return "Bot Token không đúng. Hãy sao chép lại token từ @BotFather."
    if status_code == 404:
        return "Bot Token không tồn tại. Hãy sao chép lại token từ @BotFather."
    if "chat not found" in description.lower():
        return "Không tìm thấy Chat ID. Hãy nhắn một tin cho bot trước, rồi bấm 'Lấy Chat ID tự động'."
    if status_code == 403:
        return "Bot bị chặn hoặc chưa được bắt đầu. Mở Telegram, vào bot và bấm Start."
    if status_code == 409:
        return "Bot đang dùng webhook nên không đọc được tin nhắn. Hãy nhập Chat ID thủ công."
    return f"Telegram báo lỗi: {description or f'HTTP {status_code}'}"
