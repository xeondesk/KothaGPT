"""Corpus loading for tokenizer training.

Accepts a single text file, a JSON/JSONL (gzip) file, or a directory of
documents (reuses the Phase 1A ingestion formats).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

__all__ = ["load_corpus"]

_JSONL_EXTS = (".jsonl", ".jsonl.gz")


def load_corpus(path: str | Path) -> list[str]:
    """Return the list of documents under ``path`` as plain text."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"corpus not found: {p}")
    if p.is_dir():
        from data.pipeline.io import iter_records

        texts = [record["text"] for record in iter_records(p)]
        if not texts:
            raise ValueError(f"no documents found under {p}")
        return texts
    if p.suffix == ".txt":
        return [p.read_text(encoding="utf-8")]
    if p.suffix in _JSONL_EXTS or p.name.endswith(".jsonl.gz"):
        return _read_jsonl(p)
    if p.suffix == ".json":
        return _read_json(p)
    raise ValueError(f"unsupported corpus format: {p.suffix}")


def _read_jsonl(path: Path) -> list[str]:
    texts: list[str] = []
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "text" not in record:
                raise ValueError(f"{path}:{lineno}: missing 'text' field")
            texts.append(record["text"])
    if not texts:
        raise ValueError(f"no records in {path}")
    return texts


def _read_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of objects")
    texts = [record["text"] for record in data if isinstance(record, dict)]
    if not texts:
        raise ValueError(f"no records in {path}")
    return texts
