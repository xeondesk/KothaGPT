"""Deterministic train/validation split.

The split is content-addressed: the same normalized text always lands in the
same split regardless of input order or sharding, which keeps the split
reproducible and stable across pipeline runs.
"""

from __future__ import annotations

from hashlib import sha256

__all__ = ["split_set", "split_record"]

_SPLIT_NAMES = ("train", "validation")


def split_key(text: str) -> int:
    """Deterministic 32-bit bucket derived from the text content."""
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:4], "big")


def split_set(text: str, validation_ratio: float = 0.02) -> str:
    """Return ``train`` or ``validation`` for ``text``."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")
    bucket = split_key(text)
    if bucket / 0xFFFFFFFF < validation_ratio:
        return "validation"
    return "train"


def split_record(record: dict, *, text_key: str = "text", validation_ratio: float = 0.02) -> dict:
    """Attach a ``split`` field to a record (does not mutate the input)."""
    copy = dict(record)
    copy["split"] = split_set(record[text_key], validation_ratio)
    return copy
