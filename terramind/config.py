"""Central configuration. Values come from environment variables, a local .env
file, or Streamlit secrets (in that order of precedence: secrets > env > default)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "assets" / "samples"


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _get(key: str, default: str = "") -> str:
    try:  # Streamlit secrets, if running inside Streamlit and configured
        import streamlit as st  # noqa: WPS433

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    supabase_url: str = field(default_factory=lambda: _get("SUPABASE_URL"))
    supabase_key: str = field(default_factory=lambda: _get("SUPABASE_KEY"))
    supabase_bucket: str = field(default_factory=lambda: _get("SUPABASE_BUCKET", "analysis-images"))
    gee_project: str = field(default_factory=lambda: _get("GEE_PROJECT"))
    gee_service_account: str = field(default_factory=lambda: _get("GEE_SERVICE_ACCOUNT"))
    gee_key_file: str = field(default_factory=lambda: _get("GEE_KEY_FILE"))
    # Full service-account JSON key as text (for Streamlit Cloud Secrets, where no key file can be uploaded)
    gee_key_json: str = field(default_factory=lambda: _get("GEE_SERVICE_ACCOUNT_JSON"))
    yolo_weights: str = field(default_factory=lambda: _get("YOLO_WEIGHTS", str(MODELS_DIR / "terramind_yolo.pt")))
    seg_weights: str = field(default_factory=lambda: _get("SEG_WEIGHTS", str(MODELS_DIR / "terramind_unet.pt")))
    hansen_asset: str = field(default_factory=lambda: _get("HANSEN_ASSET", "UMD/hansen/global_forest_change_2023_v1_11"))
    local_db: str = field(default_factory=lambda: _get("LOCAL_DB", str(DATA_DIR / "terramind.db")))


settings = Settings()
