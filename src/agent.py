from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Tìm thông tin từ cơ sở dữ liệu (top_k chunks gần nghĩa nhất)
        results = self.store.search(question, top_k=top_k)
        
        # 2. Xây dựng prompt chứa các dữ liệu ngữ cảnh
        context_texts = [r.get("content", "") for r in results]
        context_str = "\n\n---\n\n".join(context_texts)
        
        prompt = (
            "You are a helpful assistant. Use ONLY the given Context carefully to answer the Question.\n"
            "If the answer is not contained within the Context, say 'I don't have enough information'.\n\n"
            f"[Context]:\n{context_str}\n\n"
            f"[Question]: {question}\n\n"
            "Answer:"
        )
        
        # 3. Gửi cho LLM xử lý
        return self.llm_fn(prompt)
