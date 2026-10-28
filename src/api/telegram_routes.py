"""REST endpoints for configuring Telegram emergency notifications from the dashboard."""

from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.alerts import telegram
from src.alerts.telegram import TelegramClient, TelegramError, TelegramSettings

router = APIRouter(prefix="/api/settings/telegram", tags=["settings"])

_settings_listeners: list[Callable[[TelegramSettings], None]] = []


def add_settings_listener(callback: Callable[[TelegramSettings], None]) -> None:
    """Register a callback invoked whenever Telegram settings are saved."""
    _settings_listeners.append(callback)


class TelegramSettingsRequest(BaseModel):
    enabled: bool = True
    bot_token: Optional[str] = None  # Bỏ trống = giữ token đã lưu
    chat_id: str = ""


class TelegramTokenRequest(BaseModel):
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None


def _public_view(settings: TelegramSettings) -> dict[str, Any]:
    return {
        "enabled": settings.enabled,
        "has_token": bool(settings.bot_token),
        "token_masked": settings.masked_token(),
        "chat_id": settings.chat_id,
        "ready": settings.is_ready,
    }


def _resolve_token(candidate: Optional[str]) -> str:
    token = (candidate or "").strip() or telegram.load_telegram_settings().bot_token
    if not token:
        raise HTTPException(status_code=400, detail="Vui lòng nhập Bot Token lấy từ @BotFather.")
    if not telegram.is_valid_token(token):
        raise HTTPException(
            status_code=400,
            detail="Bot Token không đúng định dạng (ví dụ: 123456789:AAH...). Hãy sao chép lại từ @BotFather.",
        )
    return token


@router.get("")
def get_telegram_settings() -> dict[str, Any]:
    return _public_view(telegram.load_telegram_settings())


@router.post("")
def save_telegram_settings(req: TelegramSettingsRequest) -> dict[str, Any]:
    current = telegram.load_telegram_settings()
    chat_id = req.chat_id.strip()

    if req.enabled:
        token = _resolve_token(req.bot_token)
        if not chat_id:
            raise HTTPException(status_code=400, detail="Vui lòng nhập Chat ID hoặc bấm 'Lấy Chat ID tự động'.")
    else:
        token = (req.bot_token or "").strip() or current.bot_token
    if chat_id and not telegram.is_valid_chat_id(chat_id):
        raise HTTPException(status_code=400, detail="Chat ID không hợp lệ (ví dụ: 123456789 hoặc -100123456789).")

    settings = TelegramSettings(enabled=req.enabled, bot_token=token, chat_id=chat_id)
    try:
        telegram.save_telegram_settings(settings)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Không lưu được cài đặt Telegram: {e}")

    for listener in _settings_listeners:
        try:
            listener(settings)
        except Exception as e:
            print(f"[Warning] Telegram settings listener failed: {e}")

    message = "Đã lưu và BẬT thông báo Telegram." if settings.enabled else "Đã lưu. Thông báo Telegram đang TẮT."
    return {"status": "success", "message": message, "settings": _public_view(settings)}


@router.post("/detect-chat")
def detect_chat_id(req: TelegramTokenRequest) -> dict[str, Any]:
    client = TelegramClient(_resolve_token(req.bot_token))
    try:
        username = client.get_bot_username()
        chats = client.find_chats()
    except TelegramError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not chats:
        return {
            "status": "empty",
            "bot_username": username,
            "chats": [],
            "message": f"Chưa thấy tin nhắn nào. Mở Telegram, tìm @{username}, bấm Start (hoặc gửi 1 tin bất kỳ), rồi bấm lại nút này.",
        }
    return {
        "status": "success",
        "bot_username": username,
        "chats": chats,
        "message": f"Đã tìm thấy {len(chats)} cuộc trò chuyện với @{username}.",
    }


@router.post("/test")
def send_test_message(req: TelegramTokenRequest) -> dict[str, Any]:
    saved = telegram.load_telegram_settings()
    token = _resolve_token(req.bot_token)
    chat_id = (req.chat_id or "").strip() or saved.chat_id
    if not chat_id:
        raise HTTPException(status_code=400, detail="Vui lòng nhập Chat ID hoặc bấm 'Lấy Chat ID tự động'.")

    text = (
        "✅ Kết nối Telegram thành công!\n\n"
        "Đây là tin nhắn thử từ Fall & Stroke Warning System. Khi phát hiện bất thường, "
        "bạn sẽ nhận được cảnh báo kèm ảnh chụp:\n\n"
        "\"Possible medical emergency / abnormal behavior detected. Please check the person.\""
    )
    try:
        TelegramClient(token).send_message(chat_id, text)
    except TelegramError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "message": "Đã gửi tin thử. Kiểm tra Telegram trên điện thoại."}
