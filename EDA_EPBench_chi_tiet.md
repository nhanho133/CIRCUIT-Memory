# EDA chi tiết — Bộ dữ liệu EPBench (Tulving Episodic Memory Benchmark)

> Phân tích khám phá toàn bộ dataset EPBench đã tải về (figshare, CC0, ~575 MB).
> Mọi số liệu tính trực tiếp từ `episodic-memory-benchmark/epbench/data/`.
> Hình: `./figures/` · Bảng số: `./data/` · Script gốc: `episodic-memory-benchmark/{eda_dataset,benchmark_full_eda}.py`.

---

## 0. Mental model — benchmark đo gì
Episodic memory = nhớ **"cái gì xảy ra, với ai, ở đâu, khi nào"**. EPBench mô phỏng bằng cách:

```
   POOL (nguyên liệu)                EVENT = (t, s, e, c)              BOOK
 100 tên × 100 họ            ┌─ t = date     (khi nào)        mỗi event → 1 chương
 100 địa điểm               ├─ s = location (ở đâu)    ──►    LLM viết thành văn xuôi
 100 nội dung (loại sk)      ├─ e = entity   (ai)             nối ~196 chương → 1 cuốn sách
 ngày 2024–2026              └─ c = content  (cái gì)         ~100k token
```
Rồi hỏi **36 mẫu câu hỏi** truy hồi theo nhiều "manh mối" (cue), chấm bằng F1/Kendall-τ so ground-truth.

---

## 1. Quy mô tổng — 4 universe, 27 sách
| Universe | #sách | Cỡ chương | Tổng Q&A | Đặc trưng |
|---|---|---|---|---|
| **Udefault** (NYC) | 5 | 19 → **1967** (1M token) | 2,907 | claude-3-5-sonnet + gpt-4o; bản chính *"Synaptic Echoes"* 196 ch / 102,870 token |
| **Unews** (world news) | 12 | 17 → 232 | 6,581 | nhiều cỡ trung gian; nhân vật phụ dày nhất |
| **Uscifi** (sci-fi) | 10 | 20 → 268 | 5,926 | **ngày tương lai 2224–2226** |
| **UdefaultOrdered** | (chỉ `answers/`) | — | — | sách sắp theo thời gian, **dựng runtime** từ default |

→ **27 sách thật** (3 universe có `books/`) + biến thể ordered. Bảng đầy đủ 27 sách: `./data/overview.csv`.

### Cấu trúc file mỗi sách (`<universe>/books/<book_dir>/`)
| File | Nội dung |
|---|---|
| `book.json` | toàn văn (1 chuỗi ~600k ký tự) |
| `df_book_groundtruth.parquet` | **events** — `chapter, date, location, entity, content, post_entities, n_*` |
| `df_qa.parquet` | **Q&A** — `q_idx, cue, cue_completed, retrieval_type, get, correct_answer, correct_answer_chapters, n_items_correct_answer, n_chapters_correct_answer, bins_items_correct_answer` |
| `df_qa_debug_widespreadness.parquet` | thống kê phụ |
Cấp universe: `events.json`, `meta_events.json`, `paragraphs/`, `answers/.../{raw_answers,evaluated_answers,chronological_answers}`.

---

## 2. Mô hình EVENT (ground-truth)
- **Đúng 1 sự kiện chính / chương** (events/chapter = 1.00).
- `entity` = nhân vật chính (persona, 1 chuỗi); `post_entities` = nhân vật phụ (list).
- `n_date / n_location / n_entity / n_content` = chỉ số trỏ vào pool universe.

**Ví dụ (default long, 4 chương đầu):**
| ch | date | location | entity | content | post_entities |
|---|---|---|---|---|---|
| 1 | Sep 13, 2025 | Bethpage Black Course | Ezra Edwards | Parkour Workshop | Noa Middleton, Mara Ledbetter |
| 2 | Sep 22, 2026 | American Museum of Natural History | Chloe Castillo | Fashion Show | Sienna Hamrick, Reid Blunt |
| 3 | Sep 22, 2026 | Port Jefferson | Henry Reed | Photography Exhibition | Ronan Guevara, Miles Pritchett, Amira Hayes |
| 4 | May 07, 2024 | Hither Hills State Park | Zoe Brown | Karaoke Night | Alma Aultman, Alondra Wilkinson |

### Pool universe (nguyên liệu sinh event)
`first_names=100`, `last_names=100` → **10,000 tên khả dĩ**; `locations=100`; `contents=100`; ngày 2024–2026 (sci-fi: 2224–2226). Mỗi sách chỉ **dùng một phần nhỏ** pool → tái sử dụng có chủ đích.

---

## 3. Đặc trưng từng universe (sách "long")
| | default 196ch | world-news 200ch | sci-fi 189ch |
|---|---|---|---|
| token | 102,870 | 69,090 | 85,155 |
| ngày unique | 37 | 44 | 39 |
| địa điểm unique | 35 | 36 | 33 |
| nội dung unique | 34 | 37 | 35 |
| nhân vật unique | 34 | 39 | 36 |
| **tái xuất ≥2 chương** | 25/34 | 31/39 | 25/36 |
| max recurrence | **Julian Ross ×17** | Lucy Carter ×14 | **Levi Rodriguez ×20** |
| nhân vật phụ / chương | 2.7 | **4.8** | 3.9 |
| khoảng ngày | 2024–2026 | 2024–2026 | **2224–2226** |

(Bảng số: `./data/universe_summary.csv`.)

---

## 4. Phát hiện cốt lõi: RECURRENCE lệch đuôi dài
Số chương mỗi nhân vật xuất hiện (default long, xếp giảm): `[17, 15, 15, 14, 13, 13, 10, 9, 9, 8, 8, 7, 7, 6, 6, ...]`.
→ **Đây là trái tim của benchmark**: 1 nhân vật rải rác qua nhiều chương → đo khả năng **tracking nhiều lần xuất hiện** của cùng entity. Sách 1M-token có nhân vật xuất hiện tới **79 chương**.
- 🖼 `./figures/A_entity_zipf.png`, `./figures/entity_recurrence.png`
- 🖼 `./figures/events_over_time.png`, `./figures/top_locations.png`, `./figures/top_contents.png`

**Độ dài chương:** trung bình 415 từ (min 176, max 655) → `./figures/A_chapter_length.png`.

---

## 5. Cấu trúc TASK (sách default long = 686 câu)
3 trục: **`cue` × `retrieval_type` × `get`**.

### 5.1 `get` — 3 chế độ = đúng 4 task của ARTEM
| `get` | #Q | Hỏi gì | = ARTEM task |
|---|---|---|---|
| `all` | 548 | liệt kê **mọi** sự kiện khớp cue | **Partial Cue Retrieval** |
| (bin `0`) | 180 | 0 sự kiện → phải nói "không có" | **Epistemic Uncertainty** |
| `latest` | 69 | chỉ **trạng thái mới nhất** | **Recent Event Identification** |
| `chronological` | 69 | trả ngày **đúng thứ tự** | **Chronological Recall** |

### 5.2 `cue` — manh mối nào được CHO (15 tổ hợp t/s/e/c + wildcard `*`)
Nhiều nhất `(*,*,ent,*)`=207, rồi `(*,s,*,*)`=75, `(*,*,*,c)`=72, `(t,*,*,*)`=72… → càng nhiều `*` càng phải "quét" nhiều.

### 5.3 `retrieval_type` — đáp án chứa gì
Times 189 · Spaces 177 · Event contents 167 · Entities 133 · Full event details 10 · Other entities 10.

### 5.4 Answer-size (độ khó) — #sự kiện khớp
| bin | 0 | 1 | 2 | 3-5 | 6+ |
|---|---|---|---|---|---|
| #Q | 180 | 180 | 108 | 128 | 90 |
⚠️ **180/686 câu có 0 đáp án** = bẫy **hallucination** (phải abstain; bịa → F1=0).
Đáp án: trung bình 2.2 sự kiện, max 17.
- 🖼 `./figures/B_qa_by_cue.png`, `./figures/B_qa_by_bin.png`, `./figures/B_qa_scaling.png` (số câu hỏi tăng theo độ dài sách: 438→704).

---

## 6. Bức tranh ĐIỂM SỐ (từ kết quả EPBench đã lưu sẵn — đã replay khớp paper ≤0.01)
### 6.1 F1 theo answer-size bin (trung bình mọi model, `get=all`)
| bin | 0 | 1 | 2 | 3-5 | 6+ |
|---|---|---|---|---|---|
| F1 | **0.861** | 0.465 | 0.428 | 0.434 | 0.464 |
→ Giỏi nói "không có" (0.86) nhưng **recall sự kiện thật chỉ ~0.45, gần như phẳng** → **nút thắt là recall (tìm đủ, không sót), không phải hallucination.** 🖼 `./figures/D_f1_by_bin.png`.

### 6.2 Theo loại (trung bình model, `get=all`)
- **retrieval_type** khó→dễ: Event contents/Spaces/Entities/Times (~0.55) → Full event details (0.67).
- **cue** khó nhất `(t,*,ent,c)` 0.45, `(*,s,ent,*)` 0.46; dễ nhất `(t,s,*,c)` 0.67, `(*,*,*,c)` 0.65. 🖼 `./figures/D_f1_by_cue.png`.

### 6.3 Heatmap model × bin — insight đắt nhất 🖼 `./figures/D_heatmap_model_bin.png`
Reasoning models **over-abstain**: o1-mini bin-0=**0.97** nhưng bin-1=**0.05** (chọn im lặng, không nhớ nổi). gpt-4o & gemini-2-pro cân bằng nhất.

### 6.4 Long-context degradation (short→long)
Mất Simple Recall nhiều nhất: **o1-mini −0.64, o1 −0.59, o3-mini −0.52**, deepseek-r1 −0.41… 🖼 `./figures/C_degradation.png`, `./figures/C_ranking_long.png`.

### 6.5 Method (sách long): RAG-chapter >> in-context
| model | prompting | RAG-chapter | RAG-paragraph | fine-tune |
|---|---|---|---|---|
| claude-3-5-sonnet | 0.468 | **0.734** | 0.660 | – |
| gpt-4o | 0.669 | **0.745** | 0.622 | – |
→ Đơn vị nhớ nên là **"episode/chương"** (chapter > paragraph); fine-tune kém nhất (0.34). (Bảng: `./data/method_comparison.csv`.)

### 6.6 Chronological gần như chưa giải được ở long-context (%exact ≈ 0).

---

## 7. Hệ quả cho việc xây Memory System (CIRCUIT)
1. **Đơn vị nhớ = episode/chương** — RAG chapter > paragraph.
2. **Recall (không sót) là nút thắt**, không phải tránh bịa → tối ưu việc tìm đủ mọi lần xuất hiện của 1 entity.
3. **3 năng lực tách biệt**: tracking đa-lần-xuất-hiện · trạng thái mới nhất · thứ tự thời gian (chronological gần như chưa ai giải ở long-context).
4. Cẩn thận **over-abstention** (bin-0 chiếm 26% câu hỏi) — cân bằng "dám trả lời" vs "biết im lặng".
5. Dùng chính **2 điểm số** (Simple Recall + Chronological Awareness) + **5 bin** + **4 task ARTEM** làm thước đo.

---

## 8. Artifacts & tái chạy
- **Hình** (18): `./figures/` — A_* (dataset/event), B_* (tasks), C_* (degradation/ranking), D_* (difficulty/heatmap).
- **Bảng**: `./data/overview.csv` (27 sách), `universe_summary.csv`, `scores_long_default_claude_686Q.csv`, `scores_short_default_claude.csv`, `method_comparison.csv`.
- **Tái chạy**:
  ```bash
  cd ../episodic-memory-benchmark
  python eda_dataset.py          # EDA cơ bản + overview
  python benchmark_full_eda.py   # EDA đầy đủ (4 universe + tasks + scores)
  python epbench_replay.py        # replay điểm paper (no-LLM, khớp ≤0.01)
  ```
