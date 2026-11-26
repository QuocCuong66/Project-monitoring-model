"""Find network cameras on the user's own LAN ("Dò camera trong mạng" button).

Nothing about a particular home network is built in: the subnet comes from this computer's
own network interfaces at scan time, so the same installer works on any user's Wi-Fi.
A host counts as a camera when it answers on an RTSP port, or on a vendor port (EZVIZ /
Hikvision 8000, Dahua 37777, XMEye 34567...) which means "camera found, RTSP still off".
"""

import asyncio
import errno
import ipaddress
import socket
from typing import Any, Optional

import psutil

DEFAULT_DISCOVERY_CFG: dict[str, Any] = {
    "rtsp_ports": [554, 8554],
    "vendor_ports": [8000, 9010, 37777, 34567],
    "http_ports": [8080],
    "connect_timeout_ms": 1000,
    "concurrency": 256,
    "max_prefix_length": 24,
    "rtsp_probe_timeout_ms": 3000,
    "diagnose_timeout_ms": 3000,
}

_REFUSED_ERRNOS = {errno.ECONNREFUSED, getattr(errno, "WSAECONNREFUSED", 10061)}


def load_discovery_config() -> dict[str, Any]:
    """`discovery:` section of camera.yaml over the built-in defaults."""
    cfg = dict(DEFAULT_DISCOVERY_CFG)
    try:
        from src.camera.camera_settings import load_camera_config
        cfg.update(load_camera_config().get("discovery", {}) or {})
    except Exception:
        pass
    return cfg


def local_ipv4_interfaces() -> list[ipaddress.IPv4Interface]:
    """IPv4 addresses (with netmask) of this computer's network adapters that are up."""
    stats = psutil.net_if_stats()
    result: list[ipaddress.IPv4Interface] = []
    for name, addrs in psutil.net_if_addrs().items():
        if name in stats and not stats[name].isup:
            continue
        for addr in addrs:
            if addr.family != socket.AF_INET or not addr.address:
                continue
            try:
                result.append(ipaddress.IPv4Interface(f"{addr.address}/{addr.netmask or '255.255.255.255'}"))
            except ValueError:
                continue
    return result


def local_ipv4_addresses() -> set[str]:
    return {str(iface.ip) for iface in local_ipv4_interfaces()}


def lan_networks(max_prefix_length: int = 24) -> list[ipaddress.IPv4Network]:
    """Private LAN subnets to scan, each capped to /max_prefix_length around this computer's IP.

    Loopback, link-local and VPN/CGNAT ranges (Tailscale 100.64.0.0/10) are skipped: a scan
    there would be slow or reach other people's devices.
    """
    networks: list[ipaddress.IPv4Network] = []
    for iface in local_ipv4_interfaces():
        ip = iface.ip
        if ip.is_loopback or ip.is_link_local or not ip.is_private or ip in ipaddress.IPv4Network("100.64.0.0/10"):
            continue
        prefix = max(iface.network.prefixlen, max_prefix_length)
        net = ipaddress.IPv4Network(f"{ip}/{prefix}", strict=False)
        if net.num_addresses > 2 and net not in networks:
            networks.append(net)
    return networks


def tcp_port_state(host: str, port: int, timeout: float) -> str:
    """'open', 'refused' (host is up, port closed) or 'timeout' (no answer at all)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError as exc:
        return "refused" if exc.errno in _REFUSED_ERRNOS else "timeout"


async def _is_open(host: str, port: int, timeout: float, sem: asyncio.Semaphore) -> bool:
    async with sem:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True


async def _scan(hosts: list[str], ports: list[int], timeout: float, concurrency: int) -> dict[str, list[int]]:
    sem = asyncio.Semaphore(max(1, concurrency))
    pairs = [(h, p) for h in hosts for p in ports]
    results = await asyncio.gather(*(_is_open(h, p, timeout, sem) for h, p in pairs))
    found: dict[str, list[int]] = {}
    for (host, port), ok in zip(pairs, results):
        if ok:
            found.setdefault(host, []).append(port)
    return found


def scan_hosts(hosts: list[str], ports: list[int], timeout: float = 0.6, concurrency: int = 256) -> dict[str, list[int]]:
    """Open ports per host (hosts with nothing open are left out)."""
    return asyncio.run(_scan(hosts, sorted(set(ports)), timeout, concurrency))


def classify_device(ip: str, open_ports: list[int], cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Turn a host's open ports into a camera candidate for the UI, or None if it is not one."""
    ports = set(open_ports)
    rtsp_ports = [p for p in cfg["rtsp_ports"] if p in ports]
    vendor = ports & set(cfg["vendor_ports"])
    http = ports & set(cfg["http_ports"])
    if not rtsp_ports and not vendor and not http:
        return None

    if 37777 in vendor:
        brand, preset = "Dahua / Imou", "dahua"
    elif 8000 in vendor:
        brand, preset = "EZVIZ / Hikvision", ("ezviz" if 9010 in vendor else "hikvision")
    elif 34567 in vendor:
        brand, preset = "XMEye / camera Trung Quốc", "custom"
    elif rtsp_ports:
        brand, preset = "Camera IP (RTSP)", "custom"
    else:
        brand, preset = "Điện thoại / camera HTTP (MJPEG)", "http_mjpeg"

    rtsp_open = bool(rtsp_ports)
    if rtsp_open:
        status = "Sẵn sàng: cổng RTSP đang mở."
    elif vendor:
        status = ("Đã thấy camera nhưng RTSP đang TẮT. Bật RTSP / \"Xem trực tiếp qua LAN\" trong app của hãng, "
                  "bấm Lưu rồi khởi động lại camera.")
    else:
        status = "Thiết bị có cổng HTTP 8080 (thường là điện thoại chạy app IP Webcam)."
    return {
        "ip": ip,
        "open_ports": sorted(ports),
        "rtsp_open": rtsp_open,
        "rtsp_port": rtsp_ports[0] if rtsp_ports else int(cfg["rtsp_ports"][0]),
        "http_port": min(http) if http else None,
        "brand": brand,
        "preset": preset,
        "status": status,
    }


def discover_cameras(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Scan the LAN(s) this computer is on and list camera candidates (RTSP-ready first)."""
    cfg = {**DEFAULT_DISCOVERY_CFG, **(cfg or load_discovery_config())}
    networks = lan_networks(int(cfg["max_prefix_length"]))
    own = local_ipv4_addresses()
    hosts = [str(h) for net in networks for h in net.hosts() if str(h) not in own]
    ports = list(cfg["rtsp_ports"]) + list(cfg["vendor_ports"]) + list(cfg["http_ports"])
    found = scan_hosts(hosts, ports, float(cfg["connect_timeout_ms"]) / 1000.0, int(cfg["concurrency"])) if hosts else {}

    devices = [d for ip, p in found.items() if (d := classify_device(ip, p, cfg)) is not None]
    devices.sort(key=lambda d: (not d["rtsp_open"], d["http_port"] is not None and len(d["open_ports"]) == 1,
                                tuple(int(x) for x in d["ip"].split("."))))
    return {
        "networks": [str(n) for n in networks],
        "computer_ips": sorted(own),
        "scanned_hosts": len(hosts),
        "devices": devices,
    }
