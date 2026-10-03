"""Multi-channel notification dispatcher (Console, WebSocket, Webhook, Telegram)."""

import os
from pathlib import Path
from typing import Callable, List, Optional
import requests
from src.alerts.event_logger import AlertEvent
from src.alerts.telegram import TelegramClient, TelegramSettings, load_telegram_settings
from src.utils.config import get_env, get_project_root


class NotificationDispatcher:
    """Dispatches emergency alerts across configured external channels."""

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        if telegram_token and telegram_chat_id:
            self.telegram_settings = TelegramSettings(enabled=True, bot_token=telegram_token, chat_id=telegram_chat_id)
        else:
            self.telegram_settings = load_telegram_settings()
        self.webhook_url = webhook_url or get_env("WEBHOOK_URL")
        self._ws_subscribers: List[Callable[[AlertEvent], None]] = []

    def subscribe_websocket(self, callback: Callable[[AlertEvent], None]) -> None:
        """Register a WebSocket broadcast callback."""
        self._ws_subscribers.append(callback)

    def update_telegram_settings(self, settings: TelegramSettings) -> None:
        """Apply Telegram settings changed from the dashboard without restarting."""
        self.telegram_settings = settings

    def dispatch(self, event: AlertEvent) -> None:
        """Dispatch event across all active notification channels."""
        # 1. Local Console Logging
        self._log_to_console(event)

        # 2. WebSocket notification
        for subscriber in self._ws_subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print(f"[Warning] Failed to push alert to WebSocket subscriber: {e}")

        # 3. HTTP Webhook
        if self.webhook_url:
            self._send_webhook(event)

        # 4. Telegram Bot
        if self.telegram_settings.is_ready:
            self._send_telegram(event)

    def _log_to_console(self, event: AlertEvent) -> None:
        print("\n" + "=" * 70)
        print("🚨 EMERGENCY ALERT TRIGGERED 🚨")
        print(f"Event ID:   {event.event_id}")
        if event.camera_name:
            print(f"Camera:     {event.camera_name}")
        print(f"Person ID:  {event.person_id}")
        print(f"Timestamp:  {event.timestamp}")
        print(f"Risk Score: {event.risk_score:.2f}")
        print(f"Message:    {event.message}")
        print(f"Evidence:   {event.evidence}")
        if event.snapshot_path:
            print(f"Snapshot:   {event.snapshot_path}")
        print("=" * 70 + "\n")

    def _send_webhook(self, event: AlertEvent) -> None:
        try:
            import json
            from dataclasses import asdict
            payload = asdict(event)
            requests.post(self.webhook_url, json=payload, timeout=3.0)
        except Exception as e:
            print(f"[Warning] Webhook alert dispatch failed: {e}")

    def _send_telegram(self, event: AlertEvent) -> None:
        settings = self.telegram_settings
        try:
            # Camera names are typed by the user: escape Markdown so "Phòng_ngủ" cannot break the message
            camera_line = ""
            if event.camera_name:
                safe_name = "".join("\\" + ch if ch in "_*`[" else ch for ch in event.camera_name)
                camera_line = f"• *Camera:* {safe_name}\n"
            msg_text = (
                f"🚨 *POSSIBLE MEDICAL EMERGENCY DETECTED*\n\n"
                f"{camera_line}"
                f"• *Person ID:* `{event.person_id}`\n"
                f"• *Risk Score:* `{event.risk_score:.2f}`\n"
                f"• *Primary Action:* `{event.evidence.get('primary_action', 'unknown')}`\n"
                f"• *Notice:* {event.message}\n"
            )
            client = TelegramClient(settings.bot_token)
            client.send_message(settings.chat_id, msg_text, parse_mode="Markdown")

            # Upload snapshot image if available
            if event.snapshot_path:
                abs_path = get_project_root() / event.snapshot_path
                if abs_path.exists():
                    client.send_photo(settings.chat_id, abs_path, caption=f"Snapshot Event: {event.event_id}")
        except Exception as e:
            print(f"[Warning] Telegram alert dispatch failed: {e}")
