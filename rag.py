from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RagChunk:
    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class RagResult:
    chunk: RagChunk
    score: float


class RagIndex:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.vectorizer = None
        self.matrix = None
        self.chunks: List[RagChunk] = []

    def load(self) -> bool:
        vectorizer_path = os.path.join(self.index_dir, "vectorizer.joblib")
        matrix_path = os.path.join(self.index_dir, "matrix.npz")
        chunks_path = os.path.join(self.index_dir, "chunks.jsonl")

        if not (os.path.exists(vectorizer_path) and os.path.exists(matrix_path) and os.path.exists(chunks_path)):
            return False

        self.vectorizer = joblib.load(vectorizer_path)
        self.matrix = sparse.load_npz(matrix_path)
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.chunks.append(
                    RagChunk(
                        chunk_id=row["chunk_id"],
                        text=row["text"],
                        source=row.get("source", ""),
                        metadata=row.get("metadata", {}),
                    )
                )
        return True

    def retrieve(self, query: str, top_k: int, min_score: float) -> List[RagResult]:
        if not query or self.vectorizer is None or self.matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        if scores.size == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[RagResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            results.append(RagResult(chunk=self.chunks[idx], score=score))
        return results


_cached_index: Optional[RagIndex] = None


def get_rag_index(index_dir: str) -> Optional[RagIndex]:
    global _cached_index
    if _cached_index and _cached_index.index_dir == index_dir:
        return _cached_index

    index = RagIndex(index_dir)
    if index.load():
        _cached_index = index
        return index
    return None


def build_rag_context(results: List[RagResult]) -> str:
    if not results:
        return ""
    blocks = []
    for i, res in enumerate(results, start=1):
        meta = res.chunk.metadata or {}
        meta_text = ", ".join(f"{k}={v}" for k, v in meta.items() if v)
        blocks.append(
            "\n".join(
                [
                    f"[资料片段 {i}] (score={res.score:.3f})",
                    f"来源: {res.chunk.source}",
                    f"元数据: {meta_text}" if meta_text else "元数据: -",
                    res.chunk.text.strip(),
                ]
            )
        )
    return "\n\n".join(blocks)
