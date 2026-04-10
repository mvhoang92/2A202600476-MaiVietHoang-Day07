from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)
        scored_records = []
        
        for r in records:
            # Tính toán so sánh độ giống nhau bằng cosine
            score = compute_similarity(query_emb, r["embedding"])
            # Thêm metadata score vào trong báo cáo
            r_with_score = dict(r)
            r_with_score["score"] = score
            scored_records.append((score, r_with_score))
            
        # Xếp hạng từ cao tới thấp
        scored_records.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_records[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection:
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection:
            # Mock cách gọi API query của thư viện thật
            query_emb = self._embedding_fn(query)
            return self._collection.query(query_embeddings=[query_emb], n_results=top_k)
        
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k)
            
        filtered = []
        for r in self._store:
            # Lọc so sánh cặp khoá của metadata
            # match = True đại diện cho giấy tờ hợp lệ không bị chặn bởi bộ lọc
            match = True
            for k, v in metadata_filter.items():
                if r.get("metadata", {}).get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(r)
                
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_size = len(self._store)
        
        # Lọc giữ lại những file MÀ KO CHỨA doc_id cần xoá
        self._store = [
            r for r in self._store 
            if r.get("metadata", {}).get("doc_id") != doc_id and r.get("id") != doc_id
        ]
        
        # Nếu tổng dung lượng tủ hiện tại nhỏ hơn hồi nãy -> có file bị xoá thành công
        return len(self._store) < initial_size
