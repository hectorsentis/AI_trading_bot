"""Authentication helpers for the Streamlit operations dashboard.

Credentials are read from environment variables (or the repository ``.env`` via
``config`` import side effects in the dashboard). No plaintext password is ever
stored by this module. Supported hash format:

    pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_file() -> None:
    """Best-effort local .env loader without overriding process env."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except Exception:
        return


_load_dotenv_file()


@dataclass(frozen=True)
class AuthConfig:
    enabled: bool
    username: str | None
    password_hash: str | None
    session_key: str


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_auth_config() -> AuthConfig:
    return AuthConfig(
        enabled=env_bool("DASHBOARD_AUTH_ENABLED", True),
        username=os.getenv("DASHBOARD_USERNAME") or None,
        password_hash=os.getenv("DASHBOARD_PASSWORD_HASH") or None,
        session_key=os.getenv("DASHBOARD_SECRET_KEY") or "dashboard_session_authenticated",
    )


def generate_password_hash(password: str, *, iterations: int = 390_000, salt: bytes | None = None) -> str:
    if not password:
        raise ValueError("Password must not be empty")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str | None) -> bool:
    if not password or not stored_hash:
        return False
    try:
        algorithm, iterations_raw, salt_hex, hash_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def current_user(st) -> str:
    cfg = load_auth_config()
    if not cfg.enabled:
        return "local_unauthenticated"
    return str(st.session_state.get("dashboard_user") or "unknown")


def verify_user_password(password: str) -> bool:
    """Verify the configured dashboard user's password for re-auth on critical actions."""
    cfg = load_auth_config()
    if not cfg.enabled:
        return False
    return verify_password(password, cfg.password_hash)


def logout(st) -> None:
    cfg = load_auth_config()
    st.session_state.pop(cfg.session_key, None)
    st.session_state.pop("dashboard_user", None)


def require_login(st) -> str | None:
    """Render login gate and return authenticated username, or None if blocked."""
    cfg = load_auth_config()
    if not cfg.enabled:
        st.warning("Dashboard authentication is disabled. Enable DASHBOARD_AUTH_ENABLED=true before server exposure.")
        st.session_state[cfg.session_key] = True
        st.session_state["dashboard_user"] = "local_unauthenticated"
        return "local_unauthenticated"

    if not cfg.username or not cfg.password_hash:
        st.error("Dashboard authentication is enabled but credentials are not configured.")
        st.code(
            "DASHBOARD_AUTH_ENABLED=true\n"
            "DASHBOARD_USERNAME=admin\n"
            "DASHBOARD_PASSWORD_HASH=<pbkdf2 hash>\n"
            "DASHBOARD_SECRET_KEY=<random secret>",
            language="bash",
        )
        st.info("Generate a hash with: python -c \"from src.dashboard_auth import generate_password_hash; print(generate_password_hash('CHANGE_ME'))\"")
        return None

    if st.session_state.get(cfg.session_key):
        return str(st.session_state.get("dashboard_user") or cfg.username)

    st.markdown("### Secure dashboard login")
    with st.form("dashboard_login", clear_on_submit=True):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Login", use_container_width=True)
    if submitted:
        if hmac.compare_digest(username, cfg.username) and verify_password(password, cfg.password_hash):
            st.session_state[cfg.session_key] = True
            st.session_state["dashboard_user"] = cfg.username
            st.rerun()
        else:
            st.error("Invalid credentials.")
    return None
