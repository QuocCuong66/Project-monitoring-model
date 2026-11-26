"""Remote / network camera URL handling and reconnect backoff."""

import time

from src.camera.rtsp import RTSPStream, build_stream_url, mask_url, validate_stream_url


def test_credentials_with_special_chars_are_encoded() -> None:
    url = build_stream_url("rtsp://cam.ddns.net:554/stream1", "admin", "p@ss:w#rd/1")
    assert url == "rtsp://admin:p%40ss%3Aw%23rd%2F1@cam.ddns.net:554/stream1"


def test_separate_credentials_replace_inline_ones() -> None:
    url = build_stream_url("rtsp://old:secret@10.0.0.5:554/live", "admin", "new")
    assert url == "rtsp://admin:new@10.0.0.5:554/live"
    assert build_stream_url("rtsp://u:p@10.0.0.5/live", None, None) == "rtsp://u:p@10.0.0.5/live"


def test_password_is_masked() -> None:
    assert mask_url("rtsp://admin:p%40ss@cam.ddns.net:554/s") == "rtsp://admin:****@cam.ddns.net:554/s"
    assert mask_url("http://192.168.1.5:8080/video") == "http://192.168.1.5:8080/video"


def test_url_validation() -> None:
    for ok in ("rtsp://1.2.3.4/s", "rtsps://cam.example.com/s", "http://h:8080/video", "rtmp://h/live/key"):
        assert validate_stream_url(ok) is None, ok
    for bad in ("file:///C:/x.mp4", "192.168.1.5:554/stream", "rtsp:///nohost"):
        assert validate_stream_url(bad) is not None, bad


def test_rtsp_forced_over_tcp_only_for_rtsp() -> None:
    assert RTSPStream("rtsp://h/s", transport="tcp")._ffmpeg_options() == "rtsp_transport;tcp"
    assert RTSPStream("http://h/video", transport="tcp")._ffmpeg_options() == ""


def test_reconnect_backoff_grows_and_is_capped() -> None:
    s = RTSPStream("rtsp://h/s", reconnect_delay_seconds=2.0, max_reconnect_delay_seconds=10.0)
    delays = []
    for _ in range(5):
        before = time.time()
        s._schedule_retry()
        delays.append(round(s._next_retry_at - before))
    assert delays == [2, 4, 8, 10, 10]


def test_unreachable_camera_can_start_offline_and_keeps_retrying() -> None:
    s = RTSPStream("rtsp://127.0.0.1:9/none", open_timeout_ms=500, read_timeout_ms=500, reconnect_delay_seconds=60.0)
    assert s.start() is False  # interactive switch: immediate failure feedback
    s.allow_offline_start = True
    try:
        assert s.start() is True
        status = s.connection_status()
        assert status["state"] == "reconnecting" and status["reconnect_attempts"] >= 1
    finally:
        s.release()
