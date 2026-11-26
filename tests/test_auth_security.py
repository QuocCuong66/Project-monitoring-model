"""Unit tests for the Authentication & Security layer."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.auth.users import (
    hash_password,
    verify_password,
    create_user,
    authenticate,
    has_any_users,
    get_user,
    change_password,
    ROLE_ADMIN,
)
from src.auth.jwt_handler import (
    create_access_token,
    create_refresh_token,
    verify_access_token,
    verify_refresh_token,
)
from src.auth.middleware import AuthMiddleware, extract_token
from src.auth import routes as auth_routes


class TestUserManagement(unittest.TestCase):
    """Test user hashing, creation, authentication, and persistence."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.patcher = patch("src.auth.users._users_file_path", return_value=Path(self.temp_dir) / "users.json")
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hash_and_verify_password(self):
        pwd = "SecretPassword123!"
        hashed = hash_password(pwd)
        self.assertNotEqual(pwd, hashed)
        self.assertTrue(verify_password(pwd, hashed))
        self.assertFalse(verify_password("WrongPassword", hashed))

    def test_create_and_authenticate_user(self):
        self.assertFalse(has_any_users())
        user = create_user("admin_test", "MyPass123", role=ROLE_ADMIN)
        self.assertEqual(user.username, "admin_test")
        self.assertTrue(has_any_users())

        # Authenticate with correct credentials
        auth_user = authenticate("admin_test", "MyPass123")
        self.assertIsNotNone(auth_user)
        self.assertEqual(auth_user.username, "admin_test")

        # Authenticate with wrong credentials
        self.assertIsNone(authenticate("admin_test", "WrongPass"))
        self.assertIsNone(authenticate("nonexistent", "MyPass123"))

    def test_duplicate_user_rejected(self):
        create_user("admin_test", "Pass12345")
        with self.assertRaises(ValueError):
            create_user("admin_test", "OtherPass")

    def test_change_password(self):
        create_user("user1", "OldPassword123")
        self.assertTrue(change_password("user1", "OldPassword123", "NewPassword456"))
        self.assertIsNotNone(authenticate("user1", "NewPassword456"))
        self.assertIsNone(authenticate("user1", "OldPassword123"))


class TestJWTHandler(unittest.TestCase):
    """Test JWT creation, validation, and expiry."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.patcher = patch(
            "src.auth.jwt_handler._secret_key_path",
            return_value=Path(self.temp_dir) / "secret.key",
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_access_token_creation_and_verification(self):
        token = create_access_token("alice", "admin")
        self.assertIsInstance(token, str)

        payload = verify_access_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "alice")
        self.assertEqual(payload["role"], "admin")
        self.assertEqual(payload["type"], "access")

    def test_refresh_token_creation_and_verification(self):
        ref_token = create_refresh_token("alice")
        payload = verify_refresh_token(ref_token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "alice")
        self.assertEqual(payload["type"], "refresh")

        # Access token must not be accepted by verify_refresh_token
        access = create_access_token("alice", "admin")
        self.assertIsNone(verify_refresh_token(access))

    def test_invalid_token(self):
        self.assertIsNone(verify_access_token("invalid.jwt.token"))


class TestAuthAPIAndMiddleware(unittest.TestCase):
    """Integration test with FastAPI TestClient."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.p1 = patch("src.auth.users._users_file_path", return_value=Path(self.temp_dir) / "users.json")
        self.p2 = patch("src.auth.jwt_handler._secret_key_path", return_value=Path(self.temp_dir) / "secret.key")
        self.p3 = patch("src.auth.middleware._is_auth_enabled", return_value=True)
        self.p4 = patch("src.auth.routes._auth_enabled", return_value=True)
        self.p1.start()
        self.p2.start()
        self.p3.start()
        self.p4.start()
        auth_routes.login_throttle.reset()

        # Build test app
        self.app = FastAPI()
        self.app.add_middleware(AuthMiddleware)
        self.app.include_router(auth_routes.router)

        @self.app.get("/health")
        def health():
            return {"status": "ok"}

        @self.app.get("/api/protected-resource")
        def protected():
            return {"secret_data": "top_secret"}

        # Host must name this machine (anti DNS-rebinding): browse it as http://localhost
        self.client = TestClient(self.app, base_url="http://localhost")

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()
        self.p3.stop()
        self.p4.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_when_auth_disabled_all_endpoints_open(self):
        with patch("src.auth.middleware._is_auth_enabled", return_value=False):
            res = self.client.get("/api/protected-resource")
            self.assertEqual(res.status_code, 200)

    def test_public_endpoint_accessible_without_auth(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})

    def test_protected_endpoint_rejects_unauthenticated(self):
        res = self.client.get("/api/protected-resource")
        self.assertEqual(res.status_code, 401)

    def test_full_setup_login_and_access_flow(self):
        # 1. Status indicates setup required
        status_res = self.client.get("/api/auth/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertTrue(status_res.json()["setup_required"])

        # 2. First-run setup (only allowed from the machine running the system)
        with patch("src.auth.routes.is_local_request", return_value=True):
            setup_res = self.client.post("/api/auth/setup", json={
                "username": "admin",
                "password": "Password123!",
            })
        self.assertEqual(setup_res.status_code, 200)
        token = setup_res.json()["access_token"]
        self.assertTrue(bool(token))

        # 3. Access protected resource with Bearer token
        protected_res = self.client.get(
            "/api/protected-resource",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(protected_res.status_code, 200)
        self.assertEqual(protected_res.json()["secret_data"], "top_secret")

        # 4. Access protected resource with cookie fallback
        cookie_res = self.client.get(
            "/api/protected-resource",
            cookies={"access_token": token},
        )
        self.assertEqual(cookie_res.status_code, 200)

        # 5. Access protected resource with query param fallback
        query_res = self.client.get(f"/api/protected-resource?token={token}")
        self.assertEqual(query_res.status_code, 200)

        # 6. Setup blocked when user already exists
        second_setup = self.client.post("/api/auth/setup", json={
            "username": "hacker",
            "password": "HackerPass123",
        })
        self.assertEqual(second_setup.status_code, 403)

        # 7. Normal login with valid credentials
        login_res = self.client.post("/api/auth/login", json={
            "username": "admin",
            "password": "Password123!",
        })
        self.assertEqual(login_res.status_code, 200)
        new_token = login_res.json()["access_token"]
        refresh_token = login_res.json()["refresh_token"]

        # 8. Token refresh
        refresh_res = self.client.post("/api/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        self.assertEqual(refresh_res.status_code, 200)
        self.assertIn("access_token", refresh_res.json())

    def test_localhost_unauthenticated_bypass(self):
        """Localhost access must succeed without token when allow_localhost_unauthenticated is True."""
        with patch("src.auth.middleware.is_localhost_client", return_value=True):
            res = self.client.get("/api/protected-resource")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json(), {"secret_data": "top_secret"})

    def test_remote_lan_client_blocked_without_token(self):
        """Remote client from LAN without token must be rejected with 401."""
        with patch("src.auth.middleware.is_localhost_client", return_value=False):
            res = self.client.get("/api/protected-resource")
            self.assertEqual(res.status_code, 401)
            self.assertIn("LAN/Wi-Fi", res.json()["detail"])

    def test_status_endpoint_returns_localhost_and_bypass_flags(self):
        """Status endpoint indicates whether current caller is localhost."""
        with patch("src.auth.routes.load_config", return_value={"auth": {"enabled": True, "allow_localhost_unauthenticated": True}}):
            res = self.client.get("/api/auth/status")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("is_localhost", data)
            self.assertIn("can_access_without_login", data)


class TestRTSPCredentialMaskingAndSSL(unittest.TestCase):
    """Test RTSP credential protection, masking, unmasking, and SSL certificate generator."""

    def test_mask_url(self):
        from src.camera.rtsp import mask_url
        raw = "rtsp://admin:SecretPass123@192.168.1.50:554/stream1"
        masked = mask_url(raw)
        self.assertEqual(masked, "rtsp://admin:****@192.168.1.50:554/stream1")
        self.assertNotIn("SecretPass123", masked)

    def test_unmask_stream_url(self):
        from src.api.server import _unmask_stream_url, current_camera_info, state_lock
        with state_lock:
            current_camera_info["rtsp_url"] = "rtsp://admin:MyPass456@192.168.1.100:554/ch1"

        masked_input = "rtsp://admin:****@192.168.1.100:554/ch1"
        restored = _unmask_stream_url(masked_input)
        self.assertEqual(restored, "rtsp://admin:MyPass456@192.168.1.100:554/ch1")

    def test_ssl_certificate_generation(self):
        from src.utils.ssl_cert import generate_self_signed_cert
        temp_dir = tempfile.mkdtemp()
        try:
            cert_p = Path(temp_dir) / "test_cert.pem"
            key_p = Path(temp_dir) / "test_key.pem"
            generate_self_signed_cert(cert_p, key_p)
            self.assertTrue(cert_p.exists())
            self.assertTrue(key_p.exists())
            self.assertIn("BEGIN CERTIFICATE", cert_p.read_text(encoding="ascii"))
            self.assertIn("BEGIN RSA PRIVATE KEY", key_p.read_text(encoding="ascii"))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestSecurityHardening(unittest.TestCase):
    """Regression tests for the LAN takeover, CSRF, DNS-rebinding, brute-force and token-revocation fixes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.patches = [
            patch("src.auth.users._users_file_path", return_value=Path(self.temp_dir) / "users.json"),
            patch("src.auth.jwt_handler._secret_key_path", return_value=Path(self.temp_dir) / "secret.key"),
            patch("src.auth.middleware._is_auth_enabled", return_value=True),
            patch("src.auth.routes._auth_enabled", return_value=True),
        ]
        for p in self.patches:
            p.start()
        auth_routes.login_throttle.reset()

        self.app = FastAPI()
        self.app.add_middleware(AuthMiddleware)
        self.app.include_router(auth_routes.router)

        @self.app.get("/api/protected-resource")
        def protected():
            return {"secret_data": "top_secret"}

        @self.app.post("/api/system/shutdown")
        def shutdown():
            return {"status": "shutting_down"}

        self.client = TestClient(self.app, base_url="http://localhost")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        auth_routes.login_throttle.reset()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_admin(self, password: str = "Password123!") -> None:
        create_user("admin", password, role=ROLE_ADMIN)

    def _login(self, password: str = "Password123!") -> dict:
        res = self.client.post("/api/auth/login", json={"username": "admin", "password": password})
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()

    def test_lan_device_cannot_claim_the_first_admin_account(self):
        res = self.client.post("/api/auth/setup", json={"username": "attacker", "password": "hacked1234"})
        self.assertEqual(res.status_code, 403)
        self.assertFalse(has_any_users())
        status = self.client.get("/api/auth/status").json()
        self.assertTrue(status["setup_required"])
        self.assertFalse(status["setup_allowed_here"])

    def test_owner_creates_first_admin_from_this_machine(self):
        with patch("src.auth.routes.is_local_request", return_value=True):
            res = self.client.post("/api/auth/setup", json={"username": "owner", "password": "Password123!"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(has_any_users())

    def test_short_password_rejected(self):
        with patch("src.auth.routes.is_local_request", return_value=True):
            res = self.client.post("/api/auth/setup", json={"username": "owner", "password": "abc123"})
        self.assertEqual(res.status_code, 400)

    def test_dns_rebinding_host_is_rejected(self):
        with patch("src.auth.middleware.is_localhost_client", return_value=True):
            ok = self.client.get("/api/protected-resource")
            rebound = self.client.get("/api/protected-resource", headers={"Host": "evil.example:8000"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(rebound.status_code, 400)

    def test_extra_host_can_be_allowed_by_env(self):
        with patch.dict(os.environ, {"ALLOWED_HOSTS": "camera.example.com"}), \
                patch("src.auth.middleware.is_localhost_client", return_value=True):
            res = self.client.get("/api/protected-resource", headers={"Host": "camera.example.com"})
        self.assertEqual(res.status_code, 200)

    def test_cross_site_write_rejected_even_with_localhost_bypass(self):
        with patch("src.auth.middleware.is_localhost_client", return_value=True):
            evil = self.client.post("/api/system/shutdown", headers={"Origin": "http://evil.example"})
            fetch_meta = self.client.post("/api/system/shutdown", headers={"Sec-Fetch-Site": "cross-site"})
            sandboxed = self.client.post("/api/system/shutdown", headers={"Origin": "null"})
            same_origin = self.client.post("/api/system/shutdown", headers={"Origin": "http://localhost"})
            no_origin = self.client.post("/api/system/shutdown")
        self.assertEqual(evil.status_code, 403)
        self.assertEqual(fetch_meta.status_code, 403)
        self.assertEqual(sandboxed.status_code, 403)
        self.assertEqual(same_origin.status_code, 200)
        self.assertEqual(no_origin.status_code, 200)

    def test_cross_site_login_rejected(self):
        self._create_admin()
        res = self.client.post("/api/auth/login", json={"username": "admin", "password": "Password123!"},
                               headers={"Origin": "http://evil.example"})
        self.assertEqual(res.status_code, 403)

    def test_login_brute_force_is_throttled(self):
        self._create_admin()
        codes = [
            self.client.post("/api/auth/login", json={"username": "admin", "password": f"wrong{i}"}).status_code
            for i in range(6)
        ]
        self.assertEqual(codes[:5], [401] * 5)
        self.assertEqual(codes[5], 429)
        # Even the right password waits until the window passes
        right = self.client.post("/api/auth/login", json={"username": "admin", "password": "Password123!"})
        self.assertEqual(right.status_code, 429)

    def test_password_change_revokes_old_tokens(self):
        self._create_admin()
        tokens = self._login()
        auth = {"Authorization": f"Bearer {tokens['access_token']}"}
        self.assertEqual(self.client.get("/api/protected-resource", headers=auth).status_code, 200)

        res = self.client.post("/api/auth/change-password", headers=auth,
                               json={"old_password": "Password123!", "new_password": "NewPassword456"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.client.get("/api/protected-resource", headers=auth).status_code, 401)
        self.assertEqual(self.client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code, 401)
        # The device that changed the password got fresh tokens
        fresh = {"Authorization": f"Bearer {res.json()['access_token']}"}
        self.assertEqual(self.client.get("/api/protected-resource", headers=fresh).status_code, 200)

    def test_logout_revokes_tokens_on_every_device(self):
        self._create_admin()
        phone = self._login()
        laptop = self._login()
        res = self.client.post("/api/auth/logout", headers={"Authorization": f"Bearer {laptop['access_token']}"})
        self.assertEqual(res.status_code, 200)
        for device in (phone, laptop):
            auth = {"Authorization": f"Bearer {device['access_token']}"}
            self.assertEqual(self.client.get("/api/protected-resource", headers=auth).status_code, 401)

    def test_refresh_issues_working_access_token(self):
        self._create_admin()
        tokens = self._login()
        res = self.client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        self.assertEqual(res.status_code, 200)
        auth = {"Authorization": f"Bearer {res.json()['access_token']}"}
        self.assertEqual(self.client.get("/api/protected-resource", headers=auth).status_code, 200)

    def test_token_of_deleted_user_rejected(self):
        from src.auth.users import delete_user
        self._create_admin()
        tokens = self._login()
        delete_user("admin")
        auth = {"Authorization": f"Bearer {tokens['access_token']}"}
        self.assertEqual(self.client.get("/api/protected-resource", headers=auth).status_code, 401)


class TestRequestGuard(unittest.TestCase):
    """Local-machine detection must not trust loopback traffic forwarded by a proxy / tunnel."""

    @staticmethod
    def _conn(client_host: str, headers: dict):
        from starlette.requests import Request
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "method": "GET", "path": "/", "headers": raw,
                        "client": (client_host, 50000), "query_string": b""})

    def test_loopback_is_local(self):
        from src.auth.request_guard import is_local_request
        self.assertTrue(is_local_request(self._conn("127.0.0.1", {"host": "localhost:8000"})))
        self.assertTrue(is_local_request(self._conn("::1", {"host": "localhost:8000"})))

    def test_lan_and_tunnelled_clients_are_not_local(self):
        from src.auth.request_guard import is_local_request
        self.assertFalse(is_local_request(self._conn("192.168.1.20", {"host": "localhost:8000"})))
        self.assertFalse(is_local_request(self._conn("127.0.0.1", {"host": "x.ngrok.app", "x-forwarded-for": "203.0.113.9"})))
        self.assertFalse(is_local_request(self._conn("127.0.0.1", {"host": "x", "cf-connecting-ip": "203.0.113.9"})))

    def test_strip_port(self):
        from src.auth.request_guard import strip_port
        self.assertEqual(strip_port("Example.com:8000"), "example.com")
        self.assertEqual(strip_port("[::1]:8000"), "::1")
        self.assertEqual(strip_port("10.0.0.5"), "10.0.0.5")


if __name__ == "__main__":
    unittest.main()
