"""Deduplication: exact (sha256) and near-duplicate (MinHash + LSH).

Exact duplicates are removed with a content hash set. Near-duplicates are
detected with word-shingle MinHash signatures bucketed by banded LSH, which
keeps memory bounded while remaining robust to small edits.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

__all__ = [
    "deduplicate",
    "exact_duplicate_keys",
    "minhash",
    "near_duplicate_groups",
    "text_hash",
]

_RECORD_TEXT_KEY = "text"


def text_hash(text: str) -> str:
    """Content hash used for exact dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_duplicate_keys(records: Iterable[str], *, seen: set[str] | None = None):
    """Yield ``(text, is_new)`` for each input text, tracking exact dupes."""
    if seen is None:
        seen = set()
    for text in records:
        digest = text_hash(text)
        is_new = digest not in seen
        if is_new:
            seen.add(digest)
        yield text, is_new


def _hash_bytes(data: bytes) -> int:
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _shingles(text: str, k: int = 5) -> list[str]:
    words = text.split()
    if len(words) < k:
        return [" ".join(words)]
    return [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]


def minhash(
    text: str,
    *,
    num_hashes: int = 128,
    shingle_size: int = 5,
) -> list[int]:
    """Compute a MinHash signature for ``text`` (pure python, deterministic)."""
    sig = [2**63 - 1] * num_hashes
    for shingle in _shingles(text, shingle_size):
        for i in range(num_hashes):
            value = _hash_bytes(f"{i}\x00{shingle}".encode("utf-8"))
            if value < sig[i]:
                sig[i] = value
    return sig


def near_duplicate_groups(
    signatures: dict[str, list[int]],
    *,
    num_bands: int = 32,
    rows_per_band: int = 4,
    threshold: float = 0.8,
) -> list[set[str]]:
    """Return connected groups of near-duplicate keys via banded LSH.

    A pair whose signatures collide in any band is treated as a candidate and
    confirmed with an exact Jaccard estimate. Bands x rows must equal the
    signature length.
    """
    expected = num_bands * rows_per_band
    for sig in signatures.values():
        if len(sig) != expected:
            raise ValueError(f"signature length {len(sig)} != bands*rows={expected}")

    buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for key, sig in signatures.items():
        for b in range(num_bands):
            band = tuple(sig[b * rows_per_band : (b + 1) * rows_per_band])
            buckets[band].append(key)

    candidates: set[frozenset[str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add(frozenset((members[i], members[j])))

    groups: list[set[str]] = []
    parent: dict[str, str] = {key: key for key in signatures}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in candidates:
        ka, kb = tuple(pair)
        if _jaccard_estimate(signatures[ka], signatures[kb]) >= threshold:
            union(ka, kb)

    by_root: dict[str, set[str]] = defaultdict(set)
    for key in signatures:
        by_root[find(key)].add(key)
    for members in by_root.values():
        if len(members) > 1:
            groups.append(members)
    return groups


def _jaccard_estimate(a: list[int], b: list[int]) -> float:
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def deduplicate(
    records: Iterable[dict[str, Any]],
    *,
    exact: bool = True,
    near: bool = False,
    threshold: float = 0.8,
    num_hashes: int = 128,
    shingle_size: int = 5,
    num_bands: int = 32,
    rows_per_band: int = 4,
    text_key: str = _RECORD_TEXT_KEY,
    log=None,
) -> list[dict[str, Any]]:
    """Remove duplicate records; returns the deduplicated list.

    Exactly equal normalized texts are always removed when ``exact``. When
    ``near`` is enabled, records are grouped by banded MinHash LSH and all but
    the first (by input order) record per group is dropped.
    """
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        text = record[text_key]
        if exact:
            digest = text_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
        kept.append(record)

    if near and len(kept) > 1:
        signatures = {
            text_hash(record[text_key]): minhash(
                record[text_key],
                num_hashes=num_hashes,
                shingle_size=shingle_size,
            )
            for record in kept
        }
        groups = near_duplicate_groups(
            signatures,
            num_bands=num_bands,
            rows_per_band=rows_per_band,
            threshold=threshold,
        )
        drop: set[str] = set()
        for group in groups:
            for member in sorted(group)[1:]:
                drop.add(member)
        kept = [r for r in kept if text_hash(r[text_key]) not in drop]

    return kept
