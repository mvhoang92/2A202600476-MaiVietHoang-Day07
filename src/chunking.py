from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        # Tìm các điểm ngắt câu bằng Regex, Dùng () tròn trong Regex để CHILL giữ lại biến thể dấu chấm nối với text
        parts = re.split(r'(\. |\! |\? |\.\n)', text)
        sentences = []
        
        # Parts trả ra là một danh sách chập chéo: [chữ, dấu câu, chữ, dấu câu...] -> Lặp qua vòng lặp bước 2 ở đây
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i] + parts[i+1] # Cộng lại để dấu thuộc về đuôi câu
            stripped_sentence = sentence.strip()
            if stripped_sentence:  # Bỏ những câu vô nghĩa trống trơn
                sentences.append(stripped_sentence)
                
        # Ghép đoạn dư cuối mảng mà không có bất kỳ ký hiệu đuôi nào
        if len(parts) % 2 != 0:
            last_part = parts[-1].strip()
            if last_part:
               sentences.append(last_part)
               
        # Thuật toán gom nhóm các câu đã rời thành các tập chunk giới hạn size
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            # Lấy mảng con rồi dùng dâu 'cách' nối lại với nhau về dạng text
            chunk_slice = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = " ".join(chunk_slice)
            chunks.append(chunk_text)
            
        return chunks

class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # 1. Đi tìm dấu ngăn cách (separator) tốt nhất hiện đang có trong văn bản
        separator = remaining_separators[-1] if remaining_separators else ""
        next_separators = []
        for i, s in enumerate(remaining_separators):
            if s == "":
                separator = s
                break
            if s in current_text:
                separator = s
                next_separators = remaining_separators[i + 1:] # Lọc các dấu ưu tiên thấp hơn cho lần đệ quy sau
                break

        # 2. Xử lý trường hợp triệt để không có phân cách nào -> cắt cứng bằng kích thước mảng
        if not separator:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        # 3. Phân cắt mảng tạm thời
        splits = list(current_text) if separator == "" else current_text.split(separator)

        # 4. Gom các mảng đã cắt thành một khối to hơn một chút nhưng nằm vùng an toàn <= chunk_size
        final_chunks = []
        current_chunk = []
        current_len = 0

        for split in splits:
            if not current_chunk:
                temp_len = len(split)
            else:
                temp_len = current_len + len(separator) + len(split)

            if temp_len <= self.chunk_size:
                current_chunk.append(split)
                current_len = temp_len
            else:
                # Nếu mảng đang chứa vượt mốc -> Lưu chữ cũ lại, khởi động lại để xử chữ mới
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_len = 0

                # Check tiếp rẽ nhánh: Lỡ như chính một block bị xé lẻ của chữ mới lại vẫn to hơn mức cho phép?
                if len(split) > self.chunk_size:
                    if next_separators:
                        # Đệ quy! Gọi ngược lại hàm Split bằng một con dao nhỏ hơn (Separator thấp hơn)
                        final_chunks.extend(self._split(split, next_separators))
                    else:
                        # Hết đạn, cắt cứng 
                        for i in range(0, len(split), self.chunk_size):
                            final_chunks.append(split[i : i + self.chunk_size])
                else:
                    current_chunk.append(split)
                    current_len = len(split)

        # Append nốt phần mảng cuối dư chưa đóng hộp
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        
        results = {}
        for name, chunker_instance in chunkers.items():
            chunks = chunker_instance.chunk(text)
            
            num_chunks = len(chunks)
            lengths = [len(c) for c in chunks] if chunks else [0]
            
            results[name] = {
                "count": num_chunks,
                "avg_length": sum(lengths) / num_chunks if num_chunks > 0 else 0.0,
                "max_length": max(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0,
                "chunks": chunks
            }
            
        return results
