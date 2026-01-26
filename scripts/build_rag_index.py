import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class SourceDoc:
    source: str
    text: str


def read_sources(source_dir: str) -> List[SourceDoc]:
    docs: List[SourceDoc] = []
    for root, _, files in os.walk(source_dir):
        for name in files:
            if not name.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(SourceDoc(source=os.path.relpath(path, source_dir), text=f.read()))
    return docs


def split_by_headings(text: str) -> List[str]:
    pattern = re.compile(r"(第[一二三四五六七八九十0-9]+章|^#+\s|^\d+\.\s)", re.MULTILINE)
    indices = [m.start() for m in pattern.finditer(text)]
    if not indices:
        return [text]
    indices.append(len(text))
    chunks = []
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return [c for c in chunks if c]


def build_chunks(docs: List[SourceDoc], max_chars: int, overlap: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        heading_chunks = split_by_headings(doc.text)
        for h_index, block in enumerate(heading_chunks, start=1):
            small_chunks = chunk_text(block, max_chars=max_chars, overlap=overlap)
            for c_index, c in enumerate(small_chunks, start=1):
                rows.append(
                    {
                        "chunk_id": f"{doc.source}#h{h_index}-c{c_index}",
                        "source": doc.source,
                        "text": c,
                        "metadata": {"heading_index": h_index},
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/rag_sources", help="RAG sources directory")
    parser.add_argument("--index-dir", default="data/rag_index", help="Output index directory")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    docs = read_sources(args.source_dir)
    if not docs:
        raise SystemExit(f"No source docs found in {args.source_dir}")

    chunks = build_chunks(docs, max_chars=args.max_chars, overlap=args.overlap)
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)

    os.makedirs(args.index_dir, exist_ok=True)
    with open(os.path.join(args.index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    joblib.dump(vectorizer, os.path.join(args.index_dir, "vectorizer.joblib"))
    sparse.save_npz(os.path.join(args.index_dir, "matrix.npz"), matrix)

    print(f"Built index: {len(chunks)} chunks -> {args.index_dir}")


if __name__ == "__main__":
    main()
