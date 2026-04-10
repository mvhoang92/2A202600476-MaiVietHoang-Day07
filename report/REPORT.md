# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Mai Việt Hoàng 2A202600476
**Nhóm:** 08
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nó có nghĩa là 2 văn bản có hướng Vector chĩa về cùng một góc, mang đồ thị hàm số giống nhau tương đương với việc hiểu là ngữ nghĩa của chúng giống nhau bất chấp độ dài chữ.

**Ví dụ HIGH similarity:**
- Sentence A: Tôi rất thích nuôi mèo.
- Sentence B: Mèo là loài động vật quyến rũ tuyệt vời.
- Tại sao tương đồng: Đều cùng bao hàm góc nhìn yêu quý về chủ đề loài mèo.

**Ví dụ LOW similarity:**
- Sentence A: Tôi rất thích nuôi chó.
- Sentence B: Giá vàng đang trên đà lao dốc.
- Tại sao khác: Hai câu thuộc hai cụm chủ đề khác biệt xa (Thú cưng vs Đầu tư tài chính).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Trục tọa độ Euclidean đo độ dài tịnh tiến nên bị sai lệch nặng khi một câu dài đụng một câu ngắn, còn Cosine Similarity đo góc lệch (thành phần tỷ lệ) nên luôn đánh giá chính xác độ tương đồng ngữ nghĩa bất chấp chênh lệch số lượng ký tự.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 22.11
> *Đáp án:* Làm tròn lên số lượng sẽ rơi vào 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> (10000 - 100) / (500 - 100) = ceil(24.75) = 25 chunks. (Số lượng chunks tạo ra nhiều hơn). Việc overlap cao giúp giữ lại sự nối tiếp mạch ngữ cảnh và ý tứ khi phần AI chặt ngang ở giữa đoạn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Văn học Việt Nam — Truyện ngắn của Nam Cao

**Tại sao nhóm chọn domain này?**
> Nam Cao là tác giả văn học hiện thực phê phán nổi bật nhất Việt Nam với ngôn ngữ giàu cảm xúc và nhiều sự kiện cụ thể, phù hợp để kiểm tra độ chính xác của hệ thống RAG. Các nhân vật và tình tiết đặc trưng (Chí Phèo rạch mặt, Lão Hạc bán Cậu Vàng, bát cháo hành của Thị Nở) tạo ra những câu hỏi benchmark dễ kiểm định kết quả đúng/sai. Ngoài ra việc xài tài liệu tiếng Việt giúp nhóm phát hiện thêm điểm yếu của mô hình embedding phương Tây khi xử lý ngôn ngữ đặc thù.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Chí Phèo | chi_pheo.txt | ~38,000 | source, extension, chunk_idx |
| 2 | Lão Hạc | lao_hac.txt | ~11,500 | source, extension, chunk_idx |
| 3 | Đời Mặt | doi_mat.txt | ~17,500 | source, extension, chunk_idx |
| 4 | Đời Thừa | doi_thua.txt | ~13,000 | source, extension, chunk_idx |
| 5 | Một Bữa No | mot_bua_no.txt | ~10,500 | source, extension, chunk_idx |
| 6 | Trẻ Con Không Được Ăn Thịt Chó | tre_con_khong_duoc_an_thit_cho.txt | ~12,500 | source, extension, chunk_idx |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "data/chi_pheo.txt" | Cho phép lọc theo từng tác phẩm cụ thể bằng search_with_filter |
| extension | string | ".txt" | Phân loại định dạng tệp nếu có thêm PDF/MD trong tương lai |
| chunk_idx | int | 15 | Theo dõi vị trí đoạn trong tác phẩm gốc, hỗ trợ debug |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên chi_pheo.txt (~38,000 ký tự):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| chi_pheo.txt | FixedSizeChunker (`fixed_size`) | ~80 | 500 ký tự | Không — cắt ngang câu |
| chi_pheo.txt | SentenceChunker (`by_sentences`) | ~95 | 400 ký tự | Tốt hơn — ngắt đúng câu |
| chi_pheo.txt | RecursiveChunker (`recursive`) | 77 | ~494 ký tự | Tốt nhất — ngắt theo đoạn |

### Strategy Của Tôi

**Loại:** RecursiveChunker với `chunk_size=500`

**Mô tả cách hoạt động:**
> Thuật toán đệ quy rẽ nhánh theo độ ưu tiên của bảng Separators (`\n\n` → `\n` → `.` → ` `). Nhát đầu tiên bao giờ cũng cố ngắt ở ranh giới Đoạn Văn (`\n\n`) để giữ nguyên mạch cảm xúc. Khi có một đoạn văn quá dài vượt mốc `chunk_size`, thuật toán đệ quy lùi xuống cắt theo dấu chấm câu. Nếu vẫn quá dài, lùi tiếp về dấu cách. Base case là khi cục text <= chunk_size.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Truyện ngắn Nam Cao đặc trưng bởi các đoạn văn ngắn xen kẽ đối thoại với nhiều dấu xuống dòng (`\n`), đây chính là pattern mà RecursiveChunker khai thác tốt nhất. Với chunk_size=500, mỗi chunk đủ nhỏ để tập trung một sự kiện đơn trong tường thuật (Chí Phèo rút dao, Lão Hạc khóc tiếc cậu Vàng...) mà không gây nhiễu. FixedSize hay SentenceChunker sẽ vô tình chia tách nhân vật ra khỏi hành động của chính mình.

**Code snippet:**
```python
from src.chunking import RecursiveChunker

chunks = RecursiveChunker(chunk_size=500).chunk(content)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|---------------------|
| chi_pheo.txt | FixedSizeChunker(500) | ~80 | 500 | Thấp — cắt ngang câu, mất ngữ cảnh |
| chi_pheo.txt | **RecursiveChunker(500) — của tôi** | 77 | ~495 | Cao — giữ nguyên đoạn văn, score 0.586 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Hoàng) | RecursiveChunker(500) | 8/10 | Cân bằng ngữ cảnh và độ chính xác | Vẫn có thể tách nhân quả ở câu dài |
| Giang | FixedSizeChunker(500, overlap=50) | 5/10 | Đơn giản, nhanh | Cắt ngang câu gây mất nghĩa |
| Hùng | SentenceChunker(max_sentences=3) | 6/10 | Ngắt đúng câu | Chunk quá ngắn, mất ngữ cảnh dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker với chunk_size=500 phù hợp nhất cho văn học Nam Cao vì tác giả viết theo lối tường thuật có tính liên tục cao — một hành động thường kéo dài qua nhiều câu liên tiếp. Khi hỏi "Tại sao Chí Phèo rạch mặt?", cần đọc cả đoạn dẫn dắt chứ không phải chỉ 1-2 câu đơn lẻ. Strategy đệ quy ngắt đúng ranh giới đoạn nên LLM nhận được ngữ cảnh đầy đủ nhất.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng module regex `re.split` tạo nhóm tham chiếu `(\. |\! |\? |\.\n)` để vừa chia câu vừa giữ nguyên được biểu tượng kết thúc mà không làm mất. Xử lý edge case mảng array sole bị rác bằng `.strip()` vòng lặp.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy rẽ nhánh theo độ ưu tiên của bảng Separators (`\n\n` → `\n` → ` `). Khi có 1 cục chunk quá lớn vượt mốc chunkSize, máy chuyển chunk khổng lồ đó chui ngược vô lại hàm `_split()` cùng lưỡi dao cắt bé hơn. Base case là khi cục text đó đã thu bé an toàn <= chunkSize.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Add: Dùng module `_embedding_fn` để lấy Vector rồi đóng túi từ điển nhét vào biến list mảng của class. Search: Quét toán Cosine vòng lặp toàn bộ mảng lấy Score rồi `sort(reverse=True)` để lấy Top K danh sách cao điểm.

**`search_with_filter` + `delete_document`** — approach:
> Filter sẽ lọc trước khi chấm Cosine Vector để đỡ hao phí RAM cho cỗ máy. Delete hoạt động linh hoạt bằng kĩ thuật `Array Comprehension` chắt lọc lấy các file loại bỏ document có thẻ `doc_id` cần xoá.

### KnowledgeBaseAgent

**`answer`** — approach:
> Lệnh gọi `.search` kéo ra dữ liệu thật liên quan, ghép join bằng line-break `\n---\n` tạo string ngữ cảnh. Đúc kết mảng đó vào prompt chung quy định Agent *"Chỉ trả lời trong phạm vi Context"*.

### Test Results

```
============================= test session starts ==============================
...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.04s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Trời mưa rất to | Mưa rơi nặng hạt | high | Cao (Gần 1.0) | Có |
| 2 | Mùa hè nóng nực | Mùa đông lạnh giá | low | Thấp (~0) | Có |
| 3 | Tôi rất yêu bóng đá | Bóng đá là môn tôi thích nhất | high | Cao | Có |
| 4 | Trái đất quay quanh mặt trời | Gà là động vật đẻ trứng | low | Thấp (Khoảng 0) | Có |
| 5 | Tôi ghét ăn cá | Tôi cực kì thích ăn cá | low | Cao (Gần 0.8) | KHÔNG |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 5 có chỉ số giống nhau cao ngất ngưởng thay vì phải ngược lại (low), đơn giản vì vector bị thu hút bởi cùng tập entity (TÔI, CÁ, ĂN) đứng chung trong 1 trường không gian của Ẩm thực. Điều này cho thấy Embeddings chỉ nắm bắt sự hiện diện của "Chủ đề" chứ không thực sự hiểu ý thức "Phủ định" ngữ nghĩa như con người được.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries trên implementation với `RecursiveChunker(500)` + `bkai-foundation-models/vietnamese-bi-encoder` + `top_k=15` + OpenAI GPT-4o-mini.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Chí Phèo chửi ai? | Chí Phèo chửi trời, chửi đời, chửi cả làng Vũ Đại |
| 2 | Thị Nở nấu gì cho Chí Phèo? | Thị Nở nấu cháo hành |
| 3 | Chí Phèo ăn vạ ai? | Ăn vạ bá Kiến và thằng Lý Cường |
| 4 | Ai bắt cậu Vàng? | Thằng Mục và thằng Xiên (người mua chó của Lão Hạc) bắt |
| 5 | Bi kịch của Hộ trong Đời Thừa là gì? | Nhà văn mất lý tưởng vì gánh nặng cơm áo, gia đình |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Chí Phèo chửi ai | "Hắn chửi đứa chết mẹ nào đẻ ra thân hắn..." (chi_pheo.txt) | 0.586 | ✅ Có | Chí Phèo chửi số phận, chửi người đẻ ra hắn — đúng bản chất |
| 2 | Thị Nở nấu gì | "Thị nấu bỏ vào cái rổ, mang ra cho Chí Phèo" (chi_pheo.txt) | 0.495 | ✅ Có | Thị Nở nấu cháo hành — đúng |
| 3 | Chí Phèo ăn vạ ai | "Chí Phèo tức khắc đến nhà đội Tảo..." (chi_pheo.txt) | 0.501 | ✅ Có | Chí Phèo ăn vạ bố con bá Kiến, thằng Lý Cường — đúng |
| 4 | Ai bắt cậu Vàng | "Lão Hạc bán cậu Vàng..." (lao_hac.txt) | 0.421 | ✅ Có | Thằng Mục, thằng Xiên tóm cẳng sau, trói bốn chân cậu Vàng — đúng |
| 5 | Bi kịch của Hộ | "Hộ mâu thuẫn giữa lý tưởng văn chương và trách nhiệm..." (doi_thua.txt) | 0.406 | ✅ Có | Xung đột lý tưởng nghệ thuật vs gánh nặng gia đình — đúng |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Thành viên dùng `FixedSizeChunker` cho thấy rõ mặt trái của chiến lược cắt máy móc: hệ thống lấy ra mảnh giấy chính xác về mặt địa lý nhưng AI bị mù vì thiếu mạch câu dẫn giải. So sánh kết quả cùng 1 câu hỏi giữa Fixed và Recursive giúp tôi nhận ra tại sao chunking context-aware lại quan trọng hơn chunking tốc độ. Ngoài ra, bài thực tế này còn dạy tôi rằng việc chọn **đúng embedding model cho ngôn ngữ** quan trọng không kém gì chunking strategy.

**Failure Analysis (Exercise 3.5):**
> **Lỗi 1 — "Lão Hạc xin bả ở đâu?":** Dù Top-1 lấy từ đúng lão_hac.txt (score=0.357), nội dung chunk lại chỉ nói về chuyện bán vườn — không chứa chi tiết xin bả Binh Tư. LLM thiếu ngữ cảnh nên hallucinate ra câu trả lời sai "xin bả ở mụ bán rượu". Điểm score thấp (0.357) là dấu hiệu truy vấn yếu — câu hỏi dùng từ "bả" (phương ngữ) nhưng văn bản gốc dùng "thuốc chó". Giải pháp: thêm query rewriting hoặc bổ sung synonym mapping cho phương ngữ miền Nam.
>
> **Lỗi 2 — Score thấp toàn cục với câu hỏi ngữ nghĩa trừu tượng:** Câu hỏi "Bi kịch của Hộ" cho score chỉ 0.406 dù Top-1 đúng truyện. Lý do: bi kịch trải rải qua nhiều đoạn văn, không đặc (dense) tại một chunk duy nhất. Giải pháp: **re-ranking** sau retrieval bằng cross-encoder để xếp lại các chunk liên quan trước khi đưa vào LLM.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Thứ nhất, đã áp dụng rồi: thay embedding model sang `bkai-foundation-models/vietnamese-bi-encoder` — model chuyên tiếng Việt, dimension 768, giúp tất cả 5/5 benchmark queries trả về chunk đúng truyện (so với 3/5 trước đó với `all-MiniLM-L6-v2`). Thứ hai, bổ sung metadata `ten_truyen` và `nhan_vat_chinh` để kích hoạt `search_with_filter` lọc đúng tác phẩm trước khi vector search, tránh cross-story confusion. Thứ ba, dùng ChromaDB persist directory thay vì pickle RAM để không phải embedding lại từ đầu mỗi lần chạy.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
