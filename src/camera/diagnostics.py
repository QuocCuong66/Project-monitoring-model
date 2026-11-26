"""Explain why a network camera stream could not be opened (shown when "Kiểm tra" fails).

OpenCV/FFmpeg only says "could not open". This walks the connection step by step on the
user's own network: name lookup -> is the IP this computer? -> is the port open? -> does the
camera have RTSP off (vendor port answers, 554 does not)? -> RTSP DESCRIBE with the given
credentials (401 = wrong password / verification code, 404 = wrong stream path).
"""

import hashlib
import ipaddress
import os
import re
import socket
from typing import Any, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from src.camera.discovery import (
    load_discovery_config, local_ipv4_addresses, local_ipv4_interfaces, scan_hosts, tcp_port_state,
)

DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322, "rtmp": 1935, "rtmps": 443, "http": 80, "https": 443}
SCAN_HINT = "Bấm \"Dò camera trong mạng\" để tìm đúng IP của camera."


def _result(reason: str, message: str, hint: str = "") -> dict[str, Any]:
    return {"reason": reason, "message": message, "hint": hint}


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _parse_auth_params(header: str) -> dict[str, str]:
    return {k.lower(): v for k, v in re.findall(r'(\w+)\s*=\s*"?([^",]*)"?', header)}


def build_authorization(challenges: list[str], method: str, uri: str, username: str, password: str) -> Optional[str]:
    """Authorization header answering an RTSP WWW-Authenticate challenge (Digest preferred, else Basic)."""
    digest = next((c for c in challenges if c.strip().lower().startswith("digest")), None)
    if digest:
        p = _parse_auth_params(digest[len("digest"):])
        realm, nonce = p.get("realm", ""), p.get("nonce", "")
        ha1 = _md5(f"{username}:{realm}:{password}")
        ha2 = _md5(f"{method}:{uri}")
        fields = [f'username="{username}"', f'realm="{realm}"', f'nonce="{nonce}"', f'uri="{uri}"']
        if "auth" in [q.strip() for q in p.get("qop", "").split(",")]:
            cnonce, nc = os.urandom(8).hex(), "00000001"
            fields += [f'response="{_md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")}"', "qop=auth", f"nc={nc}", f'cnonce="{cnonce}"']
        else:
            fields.append(f'response="{_md5(f"{ha1}:{nonce}:{ha2}")}"')
        if p.get("opaque"):
            fields.append(f'opaque="{p["opaque"]}"')
        return "Digest " + ", ".join(fields)
    if any(c.strip().lower().startswith("basic") for c in challenges):
        import base64
        return "Basic " + base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return None


def _rtsp_request(host: str, port: int, method: str, uri: str, timeout: float,
                  authorization: Optional[str] = None) -> tuple[Optional[int], list[str]]:
    """Send one RTSP request; return (status code, WWW-Authenticate challenges)."""
    lines = [f"{method} {uri} RTSP/1.0", "CSeq: 1", "User-Agent: FallStrokeWarning", "Accept: application/sdp"]
    if authorization:
        lines.append(f"Authorization: {authorization}")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))
            data = b""
            while b"\r\n\r\n" not in data and len(data) < 65536:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError:
        return None, []
    head = data.split(b"\r\n\r\n", 1)[0].decode("utf-8", "replace").split("\r\n")
    m = re.match(r"RTSP/\d\.\d\s+(\d{3})", head[0] if head else "")
    if not m:
        return None, []
    challenges = [h.split(":", 1)[1].strip() for h in head[1:] if h.lower().startswith("www-authenticate:")]
    return int(m.group(1)), challenges


def rtsp_describe_status(url: str, host: str, port: int, timeout: float) -> Optional[int]:
    """RTSP status of DESCRIBE <url> using the URL's credentials (None = no RTSP answer)."""
    parts = urlsplit(url)
    username, password = unquote(parts.username or ""), unquote(parts.password or "")
    netloc = parts.netloc.rsplit("@", 1)[-1]
    uri = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
    status, challenges = _rtsp_request(host, port, "DESCRIBE", uri, timeout)
    if status == 401 and username and challenges:
        auth = build_authorization(challenges, "DESCRIBE", uri, username, password)
        if auth:
            status, _ = _rtsp_request(host, port, "DESCRIBE", uri, timeout, auth)
    return status


def _same_lan(ip: str) -> bool:
    addr = ipaddress.IPv4Address(ip)
    return any(addr in iface.network for iface in local_ipv4_interfaces() if iface.network.prefixlen < 32)


def diagnose_stream_failure(url: str, cfg: Optional[dict[str, Any]] = None,
                            local_addresses: Optional[set[str]] = None) -> dict[str, Any]:
    """Specific Vietnamese reason why `url` did not give video: {reason, message, hint}."""
    cfg = cfg or load_discovery_config()
    # Windows retries a refused connect for ~2 s, so telling "port closed" from "no device" needs a longer wait
    timeout = float(cfg.get("diagnose_timeout_ms", 3000)) / 1000.0
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname or ""
    try:
        port = parts.port or DEFAULT_PORTS.get(scheme, 554)
    except ValueError:
        return _result("bad_url", "Số cổng trong đường dẫn camera không hợp lệ.")

    try:
        ip = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except (socket.gaierror, IndexError, UnicodeError):
        return _result("dns_failed", f"Không tìm thấy địa chỉ \"{host}\".",
                       "Kiểm tra lại tên miền DDNS / địa chỉ IP (gõ đúng, máy tính đang có mạng).")

    own = local_addresses if local_addresses is not None else local_ipv4_addresses()
    state = tcp_port_state(ip, port, timeout)

    if state != "open":
        if ip in own:
            return _result("own_ip", f"IP {ip} là địa chỉ của chính máy tính này, không phải camera.",
                           "App của hãng camera có thể đang hiện IP cũ. " + SCAN_HINT)
        other_ports = [p for p in list(cfg.get("rtsp_ports", [])) + list(cfg.get("vendor_ports", [])) if p != port]
        open_other = scan_hosts([ip], other_ports, timeout).get(ip, []) if other_ports else []
        vendor_open = [p for p in open_other if p in cfg.get("vendor_ports", [])]
        rtsp_alt = [p for p in open_other if p in cfg.get("rtsp_ports", [])]
        if rtsp_alt and scheme in ("rtsp", "rtsps"):
            return _result("wrong_port", f"Camera {ip} không mở cổng {port} nhưng có cổng RTSP {rtsp_alt[0]}.",
                           f"Đổi cổng trong đường dẫn thành :{rtsp_alt[0]}.")
        if vendor_open and scheme in ("rtsp", "rtsps"):
            return _result(
                "rtsp_disabled",
                f"Đã thấy camera ở {ip} nhưng cổng RTSP {port} đang đóng: camera chưa bật RTSP hoặc chưa khởi động lại.",
                "Trong app của hãng (EZVIZ: Cài đặt → Cài đặt Dịch vụ Cục bộ → RTSP), bật RTSP, bấm Lưu, "
                "rồi rút điện camera 10 giây và cắm lại.",
            )
        lan_hint = "" if not ipaddress.IPv4Address(ip).is_private or _same_lan(ip) else \
            f" IP {ip} không cùng dải mạng với máy tính này ({', '.join(sorted(own)) or '?'})."
        if state == "refused":
            return _result("port_closed", f"Thiết bị ở {ip} đang bật nhưng không mở cổng {port}.",
                           "Có thể đây không phải camera hoặc sai cổng." + lan_hint + " " + SCAN_HINT)
        return _result("unreachable", f"Không có thiết bị nào trả lời ở {ip}:{port}.",
                       "Sai IP, camera đang tắt / mất Wi-Fi, hoặc máy tính và camera không cùng mạng." + lan_hint
                       + " " + SCAN_HINT)

    if scheme == "rtsp":
        status = rtsp_describe_status(url, ip, port, float(cfg.get("rtsp_probe_timeout_ms", 3000)) / 1000.0)
        if status == 401:
            if not parts.username:
                return _result("auth_required", "Camera yêu cầu tài khoản và mật khẩu.",
                               "Nhập tài khoản/mật khẩu (EZVIZ: tài khoản admin, mật khẩu là mã xác minh trên tem dưới camera).")
            return _result("auth_failed", "Sai tài khoản hoặc mật khẩu / mã xác minh.",
                           "EZVIZ: tài khoản là admin, mật khẩu là mã xác minh 6 chữ IN HOA trên tem dưới đáy camera.")
        if status == 403:
            return _result("forbidden", "Camera từ chối truy cập (có thể đang tạm khóa do nhập sai mật khẩu nhiều lần).",
                           "Chờ khoảng 30 phút hoặc khởi động lại camera rồi thử lại với mật khẩu đúng.")
        if status == 404:
            return _result("bad_path", "Đăng nhập được nhưng đường dẫn luồng (phần sau :554/) không đúng với camera này.",
                           "Chọn đúng Hãng Camera để điền mẫu đường dẫn, hoặc xem hướng dẫn RTSP của hãng.")
        if status == 200:
            return _result("stream_unreadable", "Camera trả lời và đăng nhập đúng nhưng không giải mã được hình.",
                           "Tắt \"Mã hóa hình ảnh\" trong app của hãng, hoặc thử luồng phụ (EZVIZ: /h264/ch1/sub/av_stream).")
        return _result("not_rtsp", f"Cổng {port} ở {ip} đang mở nhưng không phải dịch vụ RTSP.",
                       "Kiểm tra lại cổng RTSP của camera (thường là 554). " + SCAN_HINT)

    return _result("stream_unreadable", f"Kết nối được tới {ip}:{port} nhưng không đọc được luồng video.",
                   "Kiểm tra lại đường dẫn (phần sau số cổng) và tài khoản của camera.")
