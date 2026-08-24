"""Model registry — Postgres-backed with in-memory fallback (WS-9)."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

@dataclass
class ModelRecord:
    id: str
    name: str
    family: str = "kothagpt"
    version: str = "0.2.0"
    quant_level: str = "none"
    device_profile: str = "cpu"
    artifact_path: str = ""
    digest: str = ""
    status: str = "ready"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

class ModelRegistry:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("DATABASE_URL")
        self._mem: dict[str, ModelRecord] = {}
        self._init_defaults()

    def _init_defaults(self):
        for m in [
            ModelRecord(id="kothagpt", name="Kotha GPT Base", version="0.2.0"),
            ModelRecord(id="kothagpt-small", name="Kotha GPT Small", version="0.2.0", device_profile="cpu"),
            ModelRecord(id="kothagpt-embed", name="Kotha Embed", version="0.2.0"),
            ModelRecord(id="kothagpt-rerank", name="Kotha Rerank", version="0.2.0"),
        ]:
            self._mem[m.id] = m

    def list_models(self) -> list[ModelRecord]:
        # Try Postgres if URL present, else memory
        if self.url:
            try:
                import sqlalchemy
                from sqlalchemy import create_engine, MetaData, Table, Column, String, select
                engine = create_engine(self.url)
                meta = MetaData()
                tbl = Table("model_registry", meta, Column("id", String, primary_key=True), Column("data", String))
                # best-effort read
                with engine.connect() as conn:
                    rows = conn.execute(select(tbl.c.data)).fetchall()
                    if rows:
                        import json
                        return [ModelRecord(**json.loads(r[0])) for r in rows]
            except Exception:
                pass
        return list(self._mem.values())

    def get(self, model_id: str) -> ModelRecord | None:
        return next((m for m in self.list_models() if m.id == model_id), None)

    def register(self, rec: ModelRecord) -> ModelRecord:
        self._mem[rec.id] = rec
        if self.url:
            try:
                import json, sqlalchemy
                from sqlalchemy import create_engine, MetaData, Table, Column, String, insert
                engine = create_engine(self.url)
                meta = MetaData()
                tbl = Table("model_registry", meta, Column("id", String, primary_key=True), Column("data", String))
                with engine.begin() as conn:
                    conn.execute(insert(tbl).values(id=rec.id, data=json.dumps(asdict(rec))))
            except Exception:
                pass
        return rec

_registry: ModelRegistry | None = None

def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
