from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/lao_hac.txt",
    "data/chi_pheo.txt",
    "data/doi_mat.txt",
    "data/doi_thua.txt",
    "data/mot_bua_no.txt",
    "data/tre_con_khong_duoc_an_thit_cho.txt",
]

STORY_METADATA = {
    "lao_hac": {
        "title": "Lão Hạc",
        "characters": ["Lão Hạc", "Cậu Vàng", "Ông Giáo", "Binh Tư"],
        "tags": ["bán chó"]
    },
    "chi_pheo": {
        "title": "Chí Phèo",
        "characters": ["Chí Phèo", "Thị Nở", "Bá Kiến", "Lý Cường"],
        "tags": ["tha hóa", "làng Vũ Đại", "bát cháo hành"]
    },
    "doi_mat": {
        "title": "Đôi Mắt",
        "characters": ["Hoàng", "Độ"],
        "tags": ["trí thức", "kháng chiến"]
    },
    "doi_thua": {
        "title": "Đời Thừa",
        "characters": ["Hộ", "Từ"],
        "tags": ["văn chương", "bi kịch trí thức"]
    },
    "mot_bua_no": {
        "title": "Một Bữa No",
        "characters": ["Bà lão", "Bà phó Thụ"],
    },
    "tre_con_khong_duoc_an_thit_cho": {
        "title": "Trẻ con không được ăn thịt chó",
    }
}

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions or not path.exists():
            continue

        from src.chunking import RecursiveChunker
        content = path.read_text(encoding="utf-8")
        
        # Băm nhỏ với kích thước chunk nhỏ hơn để tăng độ chính xác tìm kiếm
        chunks = RecursiveChunker(chunk_size=500).chunk(content)
        
        # Lấy metadata tương ứng với tên file
        file_key = path.stem 
        meta = STORY_METADATA.get(file_key, {})
        
        for i, text_chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{file_key}_part_{i}",
                    content=text_chunk,
                    metadata={
                        "source": str(path),
                        "title": meta.get("title", file_key),
                        "characters": ", ".join(meta.get("characters", [])),
                        "tags": ", ".join(meta.get("tags", [])),
                        "chunk_idx": i
                    },
                )
            )
    return documents


def demo_llm(prompt: str) -> str:
    """Sử dụng OpenAI API để làm bộ não LLM thật"""
    import os
    try:
        from openai import OpenAI
    except ImportError:
        return "[LỖI] Xin hãy chạy lệnh: venv/bin/pip install openai"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-your-openai-key-here":
        return "[LỖI] Bạn chưa dán key vào biến OPENAI_API_KEY trong file .env kìa!"
    
    client = OpenAI(api_key=api_key)
    print("Đang cầu viện OpenAI phân tích nội dung truyện Nam Cao...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Xài GPT-4o-mini rẻ và siêu nhanh
            messages=[
                {"role": "system", "content": "Bạn là một giáo sư văn học phân tích ngữ nghĩa xuất sắc. Nhiệm vụ của bạn là đọc kỹ đoạn [Context] được cấp và dùng duy nhất nội dung đó để trả lời [Question] bằng Tiếng Việt một cách sâu sắc."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LỖI API] {str(e)}"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    
    import pickle
    db_file = "vector_db.pkl"
    if os.path.exists(db_file):
        print(f"\n[🚀 CACHE] Tìm thấy thư viện RAM đã đóng băng! Đang nạp lại từ ổ cứng {db_file} (Siêu tốc)...")
        with open(db_file, "rb") as f:
            store._store = pickle.load(f)
    else:
        print("\n[⏳ CACHE] Chưa có thư viện lưu trữ. Bắt đầu ép xung đọc 5 cuốn sách và gọi HuggingFace tính Vector...")
        docs = load_documents_from_files(files)
        if not docs:
            print("\nNo valid input files were loaded.")
            print("Create files matching the sample paths above, then rerun:")
            print("  python3 main.py")
            return 1
            
        print(f"\nLoaded {len(docs)} documents")
        store.add_documents(docs)
        
        # Nén toàn bộ biến RAM xuống file cứng
        with open(db_file, "wb") as f:
            pickle.dump(store._store, f)
        print(f"\n[🔥 CACHE] Bùm! Đã đúc thành công 207 mã vạch xuống file ổ cứng {db_file} để dùng mãi mãi thâu đêm!")

    print(f"\nĐang tàng trữ {store.get_collection_size()} documents trong EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=15)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=15))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
