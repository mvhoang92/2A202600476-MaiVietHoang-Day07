# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Mai Việt Hoàng
**Nhóm:** Nhóm Nam Cao
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

**Loại:** RecursiveChunker với `chunk_size=1000`

**Mô tả cách hoạt động:**
> Thuật toán đệ quy rẽ nhánh theo độ ưu tiên của bảng Separators (`\n\n` → `\n` → `.` → ` `). Nhát đầu tiên bao giờ cũng cố ngắt ở ranh giới Đoạn Văn (`\n\n`) để giữ nguyên mạch cảm xúc. Khi có một đoạn văn quá dài vượt mốc `chunk_size`, thuật toán đệ quy lùi xuống cắt theo dấu chấm câu. Nếu vẫn quá dài, lùi tiếp về dấu cách. Base case là khi cục text <= chunk_size.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Truyện ngắn Nam Cao đặc trưng bởi các đoạn văn ngắn xen kẽ đối thoại với nhiều dấu xuống dòng (`\n`), đây chính là pattern mà RecursiveChunker khai thác tốt nhất. Với chunk_size=1000, mỗi tờ giấy đủ dài để chứa trọn một sự kiện tường thuật (Chí Phèo rút dao, Lão Hạc bán chó...) mà không bị cắt đứt giữa hành động. FixedSize hay SentenceChunker sẽ vô tình chia tách nhân vật ra khỏi hành động của chính mình.

**Code snippet:**
```python
from src.chunking import RecursiveChunker

chunks = RecursiveChunker(chunk_size=1000).chunk(content)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|---------------------|
| chi_pheo.txt | FixedSizeChunker(500) | ~80 | 500 | Thấp — cắt ngang câu, mất ngữ cảnh |
| chi_pheo.txt | **RecursiveChunker(1000) — của tôi** | 77 | ~1000 | Cao — giữ nguyên đoạn văn, score 0.695 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Hoàng) | RecursiveChunker(1000) | 8/10 | Bảo toàn ngữ cảnh dài | Chunk lớn hơn tốn context window |
| Thành viên B | FixedSizeChunker(500, overlap=50) | 5/10 | Đơn giản, nhanh | Cắt ngang câu gây mất nghĩa |
| Thành viên C | SentenceChunker(max_sentences=3) | 6/10 | Ngắt đúng câu | Chunk quá ngắn, mất ngữ cảnh dài |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker với chunk_size=1000 phù hợp nhất cho văn học Nam Cao vì tác giả viết theo lối tường thuật có tính liên tục cao — một hành động thường kéo dài qua nhiều câu liên tiếp. Khi hỏi "Tại sao Chí Phèo rạch mặt?", cần đọc cả đoạn dẫn dắt chứ không phải chỉ 1-2 câu đơn lẻ. Strategy đệ quy ngắt đúng ranh giới đoạn nên LLM nhận được ngữ cảnh đầy đủ nhất.

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

Chạy 5 benchmark queries trên implementation với `RecursiveChunker(1000)` + `all-MiniLM-L6-v2` + `top_k=15` + OpenAI GPT-4o-mini.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Chí Phèo chửi ai? | Chí Phèo chửi trời, chửi đời, chửi cả làng Vũ Đại |
| 2 | Thị Nở nấu gì cho Chí Phèo? | Thị Nở nấu cháo hành |
| 3 | Chí Phèo ăn vạ ai? | Ăn vạ đội Tảo và bá Kiến |
| 4 | Ai bắt cậu Vàng? | Lão Hạc bán cậu Vàng |
| 5 | Bi kịch của Hộ trong Đời Thừa là gì? | Nhà văn mất lý tưởng vì gánh nặng cơm áo |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Chí Phèo chửi ai | "Vì thế hắn chửi… chửi như những người say rượu hát" | 0.695 | ✅ Có | Chí Phèo chửi trời và đời, cả làng Vũ Đại — đúng |
| 2 | Thị Nở nấu gì | "Bát cháo húp xong rồi, thị Nở đỡ lấy bát..." | 0.642 | ✅ Có (rank 2) | Thị Nở nấu cháo hành — đúng |
| 3 | Chí Phèo ăn vạ ai | "Hắn tức khắc đến nhà đội Tảo, cất tiếng chửi..." | 0.597 | ✅ Có | Chí Phèo ăn vạ đội Tảo — đúng |
| 4 | Ai bắt cậu Vàng | "Con chó hơi gầy. Hai bát tiết canh đông lắm..." | 0.569 | ❌ Không (sai truyện) | Không đủ thông tin |
| 5 | Bi kịch của Hộ | "Đôi lông mày rậm của Hộ nhíu lại..." | 0.598 | ⚠️ Một phần | Không đủ thông tin rõ ràng |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Thành viên dùng `FixedSizeChunker` cho thấy rõ mặt trái của chiến lược cắt máy móc: hệ thống lấy ra mảnh giấy chính xác về mặt địa lý nhưng AI bị mù vì thiếu mạch câu dẫn giải. So sánh kết quả cùng 1 câu hỏi giữa Fixed và Recursive giúp tôi nhận ra tại sao chunking context-aware lại quan trọng hơn chunking tốc độ.

**Failure Analysis (Exercise 3.5):**
> **Lỗi 1 — "Ai bắt cậu Vàng?":** Hệ thống trả về Top-1 từ truyện *Trẻ con không được ăn thịt chó* thay vì *Lão Hạc*. Nguyên nhân: mô hình `all-MiniLM-L6-v2` được huấn luyện tiếng Anh nên không hiểu "Cậu Vàng" là tên riêng — nó match theo từ khóa "chó/bắt" sang sai truyện. Giải pháp: dùng mô hình embedding tiếng Việt (`keepitreal/vietnamese-sbert`) hoặc thêm metadata `ten_truyen` để lọc đúng nguồn.
>
> **Lỗi 2 — LLM trả lời "Không đủ thông tin":** Với câu hỏi "Tại sao Chí Phèo rạch mặt?", chunk Top-1 chứa *cảnh đâm chém* nhưng *lý do* (đòi làm người lương thiện) lại bị cắt sang mảnh giấy khác. Đây là failure điển hình khi chunking tách rời nguyên nhân khỏi hệ quả. Giải pháp: tăng overlap hoặc chunk_size để ngữ cảnh nhân quả nằm cùng một chunk.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Thứ nhất, thay embedding model sang `keepitreal/vietnamese-sbert` để xử lý đúng ngữ nghĩa tiếng Việt và tên riêng. Thứ hai, bổ sung metadata `ten_truyen` và `nhan_vat_chinh` để kích hoạt `search_with_filter` lọc đúng tác phẩm trước khi search, tránh nhầm lẫn chéo giữa các truyện. Thứ ba, dùng ChromaDB persist directory thay vì RAM để không phải embedding lại từ đầu mỗi lần chạy.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **82 / 100** |
