"""LAN camera discovery and the specific "Kiểm tra" failure reasons."""

import re
import socket
import threading
from typing import Optional

from src.camera import diagnostics
from src.camera.diagnostics import build_authorization, diagnose_stream_failure
from src.camera.discovery import DEFAULT_DISCOVERY_CFG, classify_device, scan_hosts, tcp_port_state

CFG = {**DEFAULT_DISCOVERY_CFG, "connect_timeout_ms": 300, "rtsp_probe_timeout_ms": 2000, "diagnose_timeout_ms": 3000}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeServer:
    """TCP server on 127.0.0.1; with rtsp_status it answers each request like an RTSP camera."""

    def __init__(self, rtsp_status: Optional[int] = None, password: Optional[str] = None) -> None:
        self.rtsp_status = rtsp_status
        self.password = password
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                if self.rtsp_status is None:
                    continue
                conn.settimeout(2)
                try:
                    req = conn.recv(4096).decode()
                except OSError:
                    continue
                conn.sendall(self._answer(req).encode())

    def _answer(self, req: str) -> str:
        if self.password is not None:
            auth = re.search(r"Authorization: (.*)\r\n", req)
            if not auth:
                return 'RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\nWWW-Authenticate: Digest realm="cam", nonce="n1"\r\n\r\n'
            uri = req.split(" ")[1]
            expected = build_authorization(['Digest realm="cam", nonce="n1"'], "DESCRIBE", uri, "admin", self.password)
            if auth.group(1) != expected:
                return "RTSP/1.0 401 Unauthorized\r\nCSeq: 1\r\n\r\n"
        return f"RTSP/1.0 {self.rtsp_status} X\r\nCSeq: 1\r\n\r\n"

    def close(self) -> None:
        self.sock.close()


def test_port_states() -> None:
    srv = FakeServer()
    try:
        assert tcp_port_state("127.0.0.1", srv.port, 1.0) == "open"
    finally:
        srv.close()
    assert tcp_port_state("127.0.0.1", _free_port(), 3.0) == "refused"  # Windows needs ~2 s


def test_scan_hosts_reports_only_open_ports() -> None:
    srv = FakeServer()
    try:
        found = scan_hosts(["127.0.0.1"], [srv.port, _free_port()], timeout=1.0)
    finally:
        srv.close()
    assert found == {"127.0.0.1": [srv.port]}


def test_classify_ezviz_with_rtsp_off_and_on() -> None:
    off = classify_device("192.168.1.3", [8000, 9010], CFG)
    assert off["brand"] == "EZVIZ / Hikvision" and off["preset"] == "ezviz"
    assert not off["rtsp_open"] and off["rtsp_port"] == 554 and "RTSP đang TẮT" in off["status"]
    on = classify_device("192.168.1.3", [554, 8000, 9010], CFG)
    assert on["rtsp_open"] and on["rtsp_port"] == 554
    assert classify_device("192.168.1.9", [37777, 554], CFG)["preset"] == "dahua"
    assert classify_device("192.168.1.1", [], CFG) is None


def test_own_ip_is_reported() -> None:
    port = _free_port()
    diag = diagnose_stream_failure(f"rtsp://admin:X@127.0.0.1:{port}/h264", CFG, local_addresses={"127.0.0.1"})
    assert diag["reason"] == "own_ip"
    assert "chính máy tính này" in diag["message"]


def test_rtsp_disabled_when_vendor_port_answers() -> None:
    vendor = FakeServer()
    try:
        cfg = {**CFG, "rtsp_ports": [_free_port()], "vendor_ports": [vendor.port]}
        diag = diagnose_stream_failure(f"rtsp://admin:X@127.0.0.1:{cfg['rtsp_ports'][0]}/h264", cfg, local_addresses=set())
    finally:
        vendor.close()
    assert diag["reason"] == "rtsp_disabled"
    assert "chưa bật RTSP" in diag["message"]


def test_port_closed_when_nothing_answers() -> None:
    cfg = {**CFG, "rtsp_ports": [_free_port()], "vendor_ports": [_free_port()]}
    diag = diagnose_stream_failure(f"rtsp://127.0.0.1:{_free_port()}/s", cfg, local_addresses=set())
    assert diag["reason"] == "port_closed"


def test_wrong_verification_code() -> None:
    srv = FakeServer(rtsp_status=200, password="ABCDEF")
    try:
        bad = diagnose_stream_failure(f"rtsp://admin:WRONG1@127.0.0.1:{srv.port}/h264/ch1/main/av_stream", CFG, set())
        good = diagnose_stream_failure(f"rtsp://admin:ABCDEF@127.0.0.1:{srv.port}/h264/ch1/main/av_stream", CFG, set())
        missing = diagnose_stream_failure(f"rtsp://127.0.0.1:{srv.port}/h264", CFG, set())
    finally:
        srv.close()
    assert bad["reason"] == "auth_failed" and "mã xác minh" in bad["message"]
    assert good["reason"] == "stream_unreadable"  # login fine -> the problem is the video itself
    assert missing["reason"] == "auth_required"


def test_wrong_stream_path() -> None:
    srv = FakeServer(rtsp_status=404)
    try:
        diag = diagnose_stream_failure(f"rtsp://admin:X@127.0.0.1:{srv.port}/wrong", CFG, set())
    finally:
        srv.close()
    assert diag["reason"] == "bad_path"


def test_not_rtsp_service() -> None:
    srv = FakeServer()  # accepts and closes without an RTSP answer
    try:
        diag = diagnose_stream_failure(f"rtsp://127.0.0.1:{srv.port}/s", CFG, set())
    finally:
        srv.close()
    assert diag["reason"] == "not_rtsp"


def test_unknown_hostname(monkeypatch) -> None:
    def fail(*_a, **_k):
        raise socket.gaierror("no such host")
    monkeypatch.setattr(diagnostics.socket, "getaddrinfo", fail)
    assert diagnose_stream_failure("rtsp://khong-ton-tai.invalid/s", CFG, set())["reason"] == "dns_failed"


def test_digest_authorization_matches_rfc2617() -> None:
    header = build_authorization(['Digest realm="r", nonce="n"'], "DESCRIBE", "rtsp://h/s", "u", "p")
    import hashlib
    md5 = lambda t: hashlib.md5(t.encode()).hexdigest()  # noqa: E731
    assert f'response="{md5(md5("u:r:p") + ":n:" + md5("DESCRIBE:rtsp://h/s"))}"' in header
    assert build_authorization(["Basic realm=\"r\""], "DESCRIBE", "rtsp://h/s", "u", "p") == "Basic dTpw"
