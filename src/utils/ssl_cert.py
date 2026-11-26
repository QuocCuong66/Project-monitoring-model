"""SSL / TLS Certificate generator and manager for Fall & Stroke Warning System.

Generates self-signed certificates with Subject Alternative Names (SANs)
covering localhost, 127.0.0.1, and the machine's local LAN IP.
Enables HTTPS and WSS (secure WebSockets) to protect video and telemetry in transit.
"""

import datetime
import ipaddress
import os
from pathlib import Path
import socket
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from src.utils.config import get_project_root, load_config


def get_local_lan_ip() -> str:
    """Best-effort detection of local LAN IP address (offline-safe)."""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
    valid_days: int = 3650,
) -> Tuple[Path, Path]:
    """Generate self-signed x509 certificate and private key.

    Includes SANs for localhost, 127.0.0.1, ::1, and the host's LAN IP.
    """
    cert_path = cert_path.resolve()
    key_path = key_path.resolve()
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate RSA 2048 private key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )

    # 2. Build Subject and Issuer (self-signed)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "VN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Elderly Safety AI System"),
        x509.NameAttribute(NameOID.COMMON_NAME, "FallStrokeMonitor"),
    ])

    # 3. Build Subject Alternative Names
    san_list: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        x509.IPAddress(ipaddress.IPv6Address("::1")),
    ]

    lan_ip = get_local_lan_ip()
    if lan_ip and lan_ip != "127.0.0.1":
        try:
            san_list.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
        except ValueError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=valid_days))
        .add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    # 4. Write Private Key (PEM format, no password encryption for daemon)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path.write_bytes(key_pem)

    # 5. Write Certificate (PEM format)
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(cert_pem)

    return cert_path, key_path


def get_ssl_config() -> dict:
    try:
        return (load_config("security.yaml") or {}).get("ssl", {}) or {}
    except Exception:
        return {}


def is_ssl_enabled() -> bool:
    return bool(get_ssl_config().get("enabled", False))


def resolve_ssl_paths() -> Tuple[Optional[str], Optional[str]]:
    """Return (cert_path, key_path) if SSL is enabled, else (None, None).

    Automatically generates the certificate if it does not yet exist.
    """
    cfg = get_ssl_config()
    if not cfg.get("enabled", False):
        return None, None

    root = get_project_root()
    cert_rel = cfg.get("certfile", "configs/cert.pem")
    key_rel = cfg.get("keyfile", "configs/key.pem")

    cert_path = (root / cert_rel).resolve()
    key_path = (root / key_rel).resolve()

    if not cert_path.exists() or not key_path.exists():
        if cfg.get("auto_generate", True):
            print(f"[Security] Auto-generating self-signed SSL certificate at {cert_path}...")
            generate_self_signed_cert(cert_path, key_path)
            print("[Security] SSL certificate and private key generated successfully.")
        else:
            print(f"[Warning] SSL is enabled but certificate file {cert_path} not found.")
            return None, None

    return str(cert_path), str(key_path)
