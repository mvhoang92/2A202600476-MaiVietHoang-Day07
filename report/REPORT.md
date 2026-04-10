# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* 
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
> *Viết 1-2 câu:*
> Trục tọa độ Euclidean đo độ dài tịnh tiến nên bị sai lệch nặng khi một câu dài đụng một câu ngắn, còn Cosine Similarity đo góc lệnh (thành phần tỷ lệ) nên luôn đánh giá chính xác độ tương đồng ngữ nghĩa bất chấp chênh lệch số lượng ký tự.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 22.11
> *Đáp án:* Làm tròn lên số lượng sẽ rơi vào 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* 
> (10000 - 100) / (500 - 100) = ceil(24.75) = 25 chunks. (Số lượng chunks tạo ra nhiều hơn). Việc overlap cao giúp giữ lại sự nối tiếp mạch ngữ cảnh và ý tứ khi phần AI chặt ngang ở giữa đoạn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
> Sử dụng module regex `re.split` tạo nhóm tham chiếu `(\. |\! |\? |\.\n)` để vừa chia câu vừa giữ nguyên được biểu tượng kết thúc mà không làm mất. Xử lý edge case mảng array sole bị rác bằng `.strip()` vòng lặp.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
> Thuật toán đệ quy rẽ nhánh theo độ ưu tiên của bảng Separators (`\n\n` -> `\n` -> ` `). Khi có 1 cục chunk quá lớn vượt mốc chunkSize, máy chuyển chunk khổng lồ đó chui ngược vô lại hàm `_split()` cùng lưỡi dao cắt bé hơn. Base case là khi cục text đó đã thu bé an toàn <= chunkSize.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*
> Add: Dùng module `_embedding_fn` để lấy Vector rồi đóng túi từ điển nhét vào biến list mảng của class. Search: Quét toán Cosine vòng lặp toàn bộ mảng lấy Score rồi `sort(reverse=True)` để lấy Top K danh sách cao điểm.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*
> Filter sẽ lọc trước khi chấm Cosine Vector để đỡ hao phí RAM cho cỗ máy. Delete hoạt động linh hoạt băng kĩ thuật `Array Comprehension` chắt lọc lấy các file loại bỏ document có thẻ `doc_id` cần xoá.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*
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
| 2 | Mùa hè nóng nực | Mùa đông lạnh giá | low | Thấp (Hay nghịch ~0/-1)| Có |
| 3 | Tôi rất yêu bóng đá | Bóng đá là môn tôi thích nhất | high | Cao | Có |
| 4 | Trái đất quay quanh mặt trời | Gà là động vật đẻ trứng | low | Thấp (Khoảng 0) | Có |
| 5 | Tôi ghét ăn cá | Tôi cực kì thích ăn cá | low | Cao (Gần 0.8) | KHÔNG |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*
> Pair 5 có chỉ số giống nhau cao ngất ngưởng thay vì phải ngược lại (low), đơn giản vì vector bị thu hút bởi cùng tập entity (TÔI, CÁ, ĂN) đứng chung trong 1 trường không gian của Ẩm thực. Điều này cho thấy Embeddings chỉ nắm bắt sự hiện diện của "Chủ đề" chứ không thực sự hiểu ý thức "Phủ định" ngữ nghĩa như con người được.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
