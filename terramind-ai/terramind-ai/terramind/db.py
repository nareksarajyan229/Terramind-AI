"""Persistence: Supabase (Postgres + Storage) with a local SQLite fallback.

Both backends expose the same tiny API so the UI never needs to know which is active.
"""
from __future__ import annotations

import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

TABLE = "analyses"


class LocalStore:
    name = "Local SQLite"

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(self.path) as con:
            con.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                id TEXT PRIMARY KEY, created_at TEXT, source TEXT, title TEXT,
                lat REAL, lon REAL, marz TEXT, engine TEXT, area_ha REAL,
                risk_score REAL, risk_level TEXT, top_issue TEXT,
                fractions TEXT, issues TEXT, risk TEXT, recommendations TEXT, image_url TEXT)""")

    def save(self, rec: dict, image_png: bytes | None = None) -> str:
        rec = _normalise(rec)
        if image_png:
            img_dir = Path(self.path).parent / "images"
            img_dir.mkdir(exist_ok=True)
            p = img_dir / f"{rec['id']}.png"
            p.write_bytes(image_png)
            rec["image_url"] = str(p)
        cols = list(rec.keys())
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in rec.values()]
        with sqlite3.connect(self.path) as con:
            con.execute(f"INSERT OR REPLACE INTO {TABLE} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        return rec["id"]

    def list(self, limit: int = 500) -> list[dict]:
        with sqlite3.connect(self.path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(f"SELECT * FROM {TABLE} ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("fractions", "issues", "risk", "recommendations"):
                if d.get(k):
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    def delete(self, rid: str) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute(f"DELETE FROM {TABLE} WHERE id = ?", (rid,))


class SupabaseStore:
    name = "Supabase"

    def __init__(self, url: str, key: str, bucket: str):
        from supabase import create_client

        self.client = create_client(url, key)
        self.bucket = bucket
        self.client.table(TABLE).select("id").limit(1).execute()  # fail fast if schema missing

    def save(self, rec: dict, image_png: bytes | None = None) -> str:
        rec = _normalise(rec)
        if image_png:
            path = f"{rec['id']}.png"
            self.client.storage.from_(self.bucket).upload(path, image_png, {"content-type": "image/png"})
            rec["image_url"] = self.client.storage.from_(self.bucket).get_public_url(path)
        self.client.table(TABLE).upsert(rec).execute()
        return rec["id"]

    def list(self, limit: int = 500) -> list[dict]:
        res = self.client.table(TABLE).select("*").order("created_at", desc=True).limit(limit).execute()
        return res.data or []

    def delete(self, rid: str) -> None:
        self.client.table(TABLE).delete().eq("id", rid).execute()


def _normalise(rec: dict) -> dict:
    rec = dict(rec)
    rec.setdefault("id", str(uuid.uuid4()))
    rec.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    rec.setdefault("image_url", None)
    return rec


_store = None
_error = ""


def get_store():
    """Return Supabase store when configured and reachable, else local SQLite."""
    global _store, _error
    if _store is not None:
        return _store
    if settings.supabase_url and settings.supabase_key:
        try:
            _store = SupabaseStore(settings.supabase_url, settings.supabase_key, settings.supabase_bucket)
            return _store
        except Exception as exc:
            _error = f"Supabase unavailable ({str(exc)[:120]}); using local SQLite."
    _store = LocalStore(settings.local_db)
    return _store


def store_error() -> str:
    return _error


def png_bytes(rgb) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
