"""Unit tests for Telegram notification settings, client and dashboard API."""

from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.alerts import telegram
from src.alerts.event_logger import AlertEvent
from src.alerts.notification import NotificationDispatcher
from src.alerts.telegram import TelegramClient, TelegramError, TelegramSettings
from src.api import telegram_routes

VALID_TOKEN = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
TEST_CONFIG: dict[str, Any] = {"telegram": {"api_base_url": "https://tg.test", "request_timeout_seconds": 1.0}}


class FakeResponse:
    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._body


class FakeTelegramApi:
    """Records Telegram API calls and replies with canned results per method."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: dict[str, Any] = {
            "getMe": {"username": "family_alert_bot"},
            "getUpdates": [],
            "sendMessage": {"message_id": 1},
            "sendPhoto": {"message_id": 2},
        }
        self.errors: dict[str, tuple[int, str]] = {}

    def post(self, url: str, timeout: Optional[float] = None, **kwargs: Any) -> FakeResponse:
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, kwargs))
        if method in self.errors:
            status, description = self.errors[method]
            return FakeResponse({"ok": False, "description": description}, status)
        return FakeResponse({"ok": True, "result": self.results[method]})

    def methods(self) -> list[str]:
        return [method for method, _ in self.calls]


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeTelegramApi:
    api = FakeTelegramApi()
    monkeypatch.setattr(telegram.requests, "post", api.post)
    monkeypatch.setattr(telegram, "load_notification_config", lambda: TEST_CONFIG)
    return api


@pytest.fixture
def settings_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "FallStrokeWarning" / "telegram.yaml"
    monkeypatch.setattr(telegram, "get_settings_path", lambda: path)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    return path


@pytest.fixture
def api_client(fake_api: FakeTelegramApi, settings_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(telegram_routes.router)
    return TestClient(app)


def _event() -> AlertEvent:
    return AlertEvent(
        event_id="evt-1",
        event_type="fall",
        person_id=1,
        timestamp="2026-10-01T10:00:00",
        risk_score=0.91,
        evidence={"primary_action": "falling"},
        snapshot_path=None,
    )


# --- Settings storage -------------------------------------------------------

def test_settings_round_trip(settings_path: Path) -> None:
    telegram.save_telegram_settings(TelegramSettings(True, VALID_TOKEN, "42"))
    loaded = telegram.load_telegram_settings()
    assert loaded == TelegramSettings(True, VALID_TOKEN, "42")
    assert loaded.is_ready


def test_settings_fall_back_to_env(settings_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", VALID_TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    assert telegram.load_telegram_settings().is_ready


def test_no_settings_means_not_ready(settings_path: Path) -> None:
    assert not telegram.load_telegram_settings().is_ready


def test_masked_token_hides_secret() -> None:
    masked = TelegramSettings(True, VALID_TOKEN, "42").masked_token()
    assert masked.startswith("123456789:")
    assert masked.endswith(VALID_TOKEN[-4:])
    assert VALID_TOKEN.split(":")[1][:-4] not in masked


@pytest.mark.parametrize("chat_id,valid", [("42", True), ("-100123456789", True), ("@family_group", True), ("abc", False), ("", False)])
def test_chat_id_validation(chat_id: str, valid: bool) -> None:
    assert telegram.is_valid_chat_id(chat_id) is valid


# --- Client -----------------------------------------------------------------

def test_find_chats_newest_first_and_deduplicated(fake_api: FakeTelegramApi) -> None:
    fake_api.results["getUpdates"] = [
        {"message": {"chat": {"id": 42, "type": "private", "first_name": "Lan", "last_name": "Nguyen"}}},
        {"message": {"chat": {"id": -100, "type": "group", "title": "Gia dinh"}}},
        {"message": {"chat": {"id": 42, "type": "private", "first_name": "Lan"}}},
    ]
    chats = TelegramClient(VALID_TOKEN).find_chats()
    assert [c["chat_id"] for c in chats] == ["42", "-100"]
    assert chats[1]["name"] == "Gia dinh"


def test_client_maps_unauthorized_to_friendly_error(fake_api: FakeTelegramApi) -> None:
    fake_api.errors["getMe"] = (401, "Unauthorized")
    with pytest.raises(TelegramError, match="Bot Token không đúng"):
        TelegramClient(VALID_TOKEN).get_bot_username()


# --- Dispatcher -------------------------------------------------------------

def test_dispatcher_sends_non_diagnostic_alert(fake_api: FakeTelegramApi, settings_path: Path) -> None:
    dispatcher = NotificationDispatcher()
    dispatcher.update_telegram_settings(TelegramSettings(True, VALID_TOKEN, "42"))
    dispatcher.dispatch(_event())

    assert fake_api.methods() == ["sendMessage"]
    sent = fake_api.calls[0][1]["data"]
    assert sent["chat_id"] == "42"
    assert "Please check the person." in sent["text"]


def test_dispatcher_names_the_camera(fake_api: FakeTelegramApi, settings_path: Path) -> None:
    dispatcher = NotificationDispatcher()
    dispatcher.update_telegram_settings(TelegramSettings(True, VALID_TOKEN, "42"))
    event = _event()
    event.camera_name = "Phòng_ngủ *2*"
    dispatcher.dispatch(event)

    text = fake_api.calls[0][1]["data"]["text"]
    assert "Camera:* Phòng\\_ngủ \\*2\\*" in text  # Markdown in the user's camera name is escaped


def test_dispatcher_skips_telegram_when_disabled(fake_api: FakeTelegramApi, settings_path: Path) -> None:
    dispatcher = NotificationDispatcher()
    dispatcher.update_telegram_settings(TelegramSettings(False, VALID_TOKEN, "42"))
    dispatcher.dispatch(_event())
    assert fake_api.calls == []


# --- Dashboard API ----------------------------------------------------------

def test_get_settings_never_returns_token(api_client: TestClient) -> None:
    telegram.save_telegram_settings(TelegramSettings(True, VALID_TOKEN, "42"))
    body = api_client.get("/api/settings/telegram").json()
    assert VALID_TOKEN not in str(body)
    assert body["has_token"] and body["ready"]


def test_save_settings_persists_and_notifies_listener(api_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[TelegramSettings] = []
    monkeypatch.setattr(telegram_routes, "_settings_listeners", [received.append])

    res = api_client.post("/api/settings/telegram", json={"enabled": True, "bot_token": VALID_TOKEN, "chat_id": "42"})

    assert res.status_code == 200
    assert telegram.load_telegram_settings() == TelegramSettings(True, VALID_TOKEN, "42")
    assert received == [TelegramSettings(True, VALID_TOKEN, "42")]


def test_save_with_blank_token_keeps_saved_token(api_client: TestClient) -> None:
    telegram.save_telegram_settings(TelegramSettings(True, VALID_TOKEN, "42"))
    res = api_client.post("/api/settings/telegram", json={"enabled": True, "bot_token": "", "chat_id": "-100"})
    assert res.status_code == 200
    assert telegram.load_telegram_settings() == TelegramSettings(True, VALID_TOKEN, "-100")


def test_save_rejects_malformed_token(api_client: TestClient) -> None:
    res = api_client.post("/api/settings/telegram", json={"enabled": True, "bot_token": "not-a-token", "chat_id": "42"})
    assert res.status_code == 400
    assert not telegram.get_settings_path().exists()


def test_save_enabled_requires_chat_id(api_client: TestClient) -> None:
    res = api_client.post("/api/settings/telegram", json={"enabled": True, "bot_token": VALID_TOKEN, "chat_id": ""})
    assert res.status_code == 400


def test_detect_chat_without_messages_explains_next_step(api_client: TestClient) -> None:
    body = api_client.post("/api/settings/telegram/detect-chat", json={"bot_token": VALID_TOKEN}).json()
    assert body["status"] == "empty"
    assert "@family_alert_bot" in body["message"]


def test_detect_chat_returns_chats(api_client: TestClient, fake_api: FakeTelegramApi) -> None:
    fake_api.results["getUpdates"] = [{"message": {"chat": {"id": 42, "type": "private", "first_name": "Lan"}}}]
    body = api_client.post("/api/settings/telegram/detect-chat", json={"bot_token": VALID_TOKEN}).json()
    assert body["chats"] == [{"chat_id": "42", "name": "Lan", "type": "private"}]


def test_send_test_message_uses_form_values(api_client: TestClient, fake_api: FakeTelegramApi) -> None:
    res = api_client.post("/api/settings/telegram/test", json={"bot_token": VALID_TOKEN, "chat_id": "42"})
    assert res.status_code == 200
    assert fake_api.methods() == ["sendMessage"]
    assert fake_api.calls[0][1]["data"]["chat_id"] == "42"


def test_send_test_message_reports_telegram_error(api_client: TestClient, fake_api: FakeTelegramApi) -> None:
    fake_api.errors["sendMessage"] = (400, "Bad Request: chat not found")
    res = api_client.post("/api/settings/telegram/test", json={"bot_token": VALID_TOKEN, "chat_id": "42"})
    assert res.status_code == 400
    assert "Chat ID" in res.json()["detail"]
