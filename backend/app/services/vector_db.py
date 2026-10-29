"""
ChronoVideo - Vector Database & Embedding Service
Manages ChromaDB persistent vector collections and semantic similarity retrieval.
"""

import math
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..config import CHROMA_DIR
from .ml_clustering import tokenize_and_vectorize, cosine_similarity

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class ChronoVectorStore:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or CHROMA_DIR
        self.client = None
        self.memory_store = {}  # In-memory fallback if ChromaDB is initializing

        if chromadb:
            try:
                self.client = chromadb.PersistentClient(path=str(self.storage_dir))
            except Exception as e:
                print(f"[ChronoVideo] ChromaDB persistent init notice: {e}. Using resilient hybrid store.")

    def get_or_create_collection(self, video_id: str):
        """Returns ChromaDB collection for the given video."""
        safe_name = f"video_{video_id}".replace("-", "_")
        if self.client:
            try:
                return self.client.get_or_create_collection(
                    name=safe_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
        return None

    def index_video_chunks(self, video_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Indexes temporal segments into ChromaDB with dense embeddings and temporal metadata."""
        if not chunks:
            return 0

        # Always maintain memory cache for instant sub-millisecond retrieval
        self.memory_store[video_id] = [
            {
                **c,
                "vector": tokenize_and_vectorize(c.get("text", ""))
            }
            for c in chunks
        ]

        collection = self.get_or_create_collection(video_id)
        if collection:
            try:
                ids = [c["chunk_id"] for c in chunks]
                documents = [c["text"] for c in chunks]
                metadatas = [
                    {
                        "start_sec": float(c["start_sec"]),
                        "end_sec": float(c["end_sec"]),
                        "time_str": c["time_str"],
                        "video_id": video_id
                    }
                    for c in chunks
                ]
                # Embeddings
                embeddings = [tokenize_and_vectorize(doc) for doc in documents]
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                return len(ids)
            except Exception as e:
                print(f"[ChronoVideo] Upsert fallback to memory index: {e}")

        return len(chunks)

    def search_similar_moments(self, video_id: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Semantic Similarity Retrieval:
        Finds the top-K temporal chunks most relevant to the user query."""
        collection = self.get_or_create_collection(video_id)
        query_vec = tokenize_and_vectorize(query)

        # 1. Try ChromaDB Native HNSW search
        if collection:
            try:
                results = collection.query(
                    query_embeddings=[query_vec],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                if results and results.get("ids") and len(results["ids"][0]) > 0:
                    matched_chunks = []
                    for i in range(len(results["ids"][0])):
                        cid = results["ids"][0][i]
                        doc = results["documents"][0][i]
                        meta = results["metadatas"][0][i]
                        dist = results["distances"][0][i] if results.get("distances") else 0.2
                        similarity = max(0.0, min(1.0, 1.0 - dist))

                        matched_chunks.append({
                            "chunk_id": cid,
                            "text": doc,
                            "start_sec": meta["start_sec"],
                            "end_sec": meta["end_sec"],
                            "time_str": meta["time_str"],
                            "similarity": round(similarity, 3)
                        })
                    return matched_chunks
            except Exception as e:
                print(f"[ChronoVideo] ChromaDB query notice: {e}. Executing vector similarity search.")

        # 2. Resilient In-Memory Vector Search Fallback
        stored = self.memory_store.get(video_id, [])
        scored = []
        for item in stored:
            sim = cosine_similarity(query_vec, item["vector"])
            scored.append({
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
                "time_str": item["time_str"],
                "similarity": round(sim, 3)
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]


# Global singleton vector store
vector_store = ChronoVectorStore()
