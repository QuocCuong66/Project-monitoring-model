"""Update API (Settings -> "Cập nhật phiên bản").

    GET  /api/update/status    - current version, cached latest release, download progress
    POST /api/update/check     - ask GitHub now (admin)
    POST /api/update/download  - download the installer in the background (admin)
    POST /api/update/install   - install it and restart the app (admin, installed app only)
"""

import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.auth.middleware import require_admin
from src.update.updater import get_updater

router = APIRouter(prefix="/api/update", tags=["update"])


@router.get("/status")
def update_status() -> dict[str, Any]:
    return get_updater().status()


@router.post("/check", dependencies=[Depends(require_admin)])
def check_for_update() -> dict[str, Any]:
    return get_updater().check(force=True)


@router.post("/download", dependencies=[Depends(require_admin)])
def download_update() -> dict[str, Any]:
    try:
        return get_updater().download()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/install", dependencies=[Depends(require_admin)])
def install_update() -> dict[str, Any]:
    try:
        return get_updater().install()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def check_in_background() -> None:
    """One quiet check at startup so the dashboard can show "new version" without waiting."""
    updater = get_updater()
    if updater.cfg.enabled and updater.cfg.check_on_startup:
        threading.Thread(target=updater.check, daemon=True, name="UpdateCheck").start()
