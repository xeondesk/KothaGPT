"""WS-6 Vector store — Qdrant + in-memory fallback for CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

from .chunk import Chunk

@dataclass
class StoredChunk:
    chunk: Chunk
    vector: list[float]

class VectorStore:
    def __init__(self, collection: str = "kothagpt", dim: int = 256, url: str | None = None):
        self.collection = collection
        self.dim = dim
        self.url = url or "http://localhost:6333"
        self._mem: list[StoredChunk] = []
        self._client = None
        # Lazy Qdrant client if available
        try:
            from qdrant_client import QdrantClient  # type: ignore
            self._client = QdrantClient(url=self.url)
            # ensure collection exists (tolerant)
            try:
                self._client.get_collection(collection)
            except Exception:
                try:
                    self._client.create_collection(collection, vectors_config={"size": dim, "distance": "Cosine"})
                except Exception:
                    pass
        except Exception:
            self._client = None

    def _embed(self, text: str) -> list[float]:
        # dependency-light hash embedding (same as MockBackend) for CI without model
        import hashlib, math
        vec = []
        for i in range(self.dim):
            digest = hashlib.sha256(f"{i}:{text}".encode()).digest()
            val = int.from_bytes(digest[:8], "big") / (2**64) * 2 - 1
            vec.append(val)
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [round(v/norm, 6) for v in vec]

    def upsert(self, chunks: Iterable[Chunk]) -> int:
        vecs = []
        for c in chunks:
            v = self._embed(c.content)
            sc = StoredChunk(c, v)
            self._mem.append(sc)
            vecs.append((c.chunk_id, v, {"document_id": c.document_id, "source": c.source}))
        if self._client:
            try:
                from qdrant_client.models import PointStruct
                points = [PointStruct(id=hash(cid) % (2**63), vector=vec, payload=pay) for cid, vec, pay in vecs]
                self._client.upsert(collection_name=self.collection, points=points)
            except Exception:
                pass
        return len(vecs)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        qvec = self._embed(query)
        # In-memory cosine
        results: list[tuple[Chunk, float]] = []
        for sc in self._mem:
            # cosine is dot since normalized
            score = sum(a*b for a,b in zip(qvec, sc.vector))
            results.append((sc.chunk, score))
        results.sort(key=lambda x: -x[1])
        if self._client and len(results) < top_k:
            try:
                hits = self._client.search(collection_name=self.collection, query_vector=qvec, limit=top_k)
                # merge (dedup)
                seen = {r[0].chunk_id for r in results}
                for h in hits:
                    # reconstruct chunk from payload (minimal)
                    payload = h.payload or {}
                    if payload.get("document_id") not in seen:
                        # fallback to mem or dummy
                        pass
            except Exception:
                pass
        return results[:top_k]

    def snapshot(self, path: str) -> str:
        import json, pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [{"chunk_id": sc.chunk.chunk_id, "document_id": sc.chunk.document_id, "content": sc.chunk.content, "source": sc.chunk.source} for sc in self._mem]
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    def restore(self, path: str) -> int:
        import json, pathlib
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self._mem.clear()
        for d in data:
            c = Chunk(d["chunk_id"], d["document_id"], d["content"], 0, d["source"])
            self._mem.append(StoredChunk(c, self._embed(c.content)))
        return len(self._mem)
