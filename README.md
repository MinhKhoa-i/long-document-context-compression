# Context Compression cho Tài liệu Dài trong LLM

> **Đồ án:** Implement Context Compression for Long Documents in LLMs

## 1. Giới thiệu đề tài

Hệ thống **Retrieval-Augmented Generation (RAG)** đang được sử dụng rộng rãi để trả lời câu hỏi dựa trên tài liệu. Tuy nhiên, khi tài liệu nguồn quá dài (ví dụ: bài báo khoa học 10-50 trang), pipeline RAG truyền thống gặp nhiều vấn đề:

- **Token quá nhiều:** Đưa toàn bộ context vào LLM tốn chi phí và chạm giới hạn context window
- **Thông tin nhiễu:** Nhiều chunk được retrieve nhưng không liên quan trực tiếp đến câu hỏi
- **Trùng lặp nội dung:** Các chunk gần nhau chứa thông tin giống nhau do overlapping
- **Lost-in-the-middle:** LLM có xu hướng bỏ sót thông tin nằm ở giữa context dài

Đồ án này xây dựng một **research pipeline** để chứng minh rằng **context compression** có thể giảm đáng kể số token đầu vào và latency, đồng thời **giữ được chất lượng câu trả lời** so với baseline không nén.

## 2. Vấn đề Long-Context trong LLM

### Tại sao long-context là vấn đề?

| Vấn đề | Mô tả |
|--------|-------|
| **Context window giới hạn** | Hầu hết LLM có giới hạn 4K-128K tokens |
| **Chi phí tuyến tính** | Cost tỉ lệ thuận với số token đầu vào |
| **Latency tăng** | Context dài → thời gian inference tăng |
| **Chất lượng giảm** | Thông tin quan trọng bị "chìm" giữa noise |
| **Lost-in-the-middle** | LLM ưu tiên thông tin ở đầu và cuối context |

### Context Compression giải quyết như thế nào?

**Context compression** là quá trình **chọn lọc và giữ lại** chỉ những phần thông tin **liên quan nhất** đến câu hỏi, loại bỏ nội dung thừa và trùng lặp, trong một **ngân sách token** cố định.

```
Trước compression:  [chunk1] [chunk2] [chunk3] [chunk4] [chunk5]  →  2500 tokens
Sau compression:    [sent3] [sent7] [sent12] [sent1]              →   800 tokens
                                                                     ↓ 68% reduction
```

## 3. Mục tiêu

1. Xây dựng pipeline QA end-to-end cho tài liệu khoa học dài (PDF)
2. Implement và so sánh **5 phương pháp compression** khác nhau
3. Đánh giá định lượng: token reduction, latency, answer quality
4. Chứng minh compression giữ được chất lượng câu trả lời

**Câu hỏi nghiên cứu:**

> Context compression có thể giảm token đầu vào và latency mà không làm giảm chất lượng câu trả lời khi thực hiện QA trên tài liệu khoa học dài?

## 4. Kiến trúc hệ thống

### Cấu trúc thư mục

```
project/
├── main.py                     # Entry point (interactive / benchmark / plot / summary)
├── requirements.txt            # Dependencies
├── .gitignore
├── data/
│   ├── papers/                 # PDF papers (input)
│   └── questions.json          # Bộ câu hỏi đánh giá
├── src/
│   ├── models.py               # Data structures (RetrievedChunk)
│   ├── pdf_loader.py           # Trích xuất text từ PDF (PyMuPDF)
│   ├── chunking.py             # Chia text thành chunks có overlap
│   ├── retrieval.py            # Semantic embedding retrieval
│   ├── compressor.py           # 5 phương pháp compression
│   ├── generator.py            # LLM answer generation (Ollama)
│   ├── evaluation.py           # Metrics đánh giá
│   ├── benchmark.py            # Chạy thí nghiệm tự động
│   ├── visualization.py        # Vẽ biểu đồ kết quả
│   └── pipeline.py             # Pipeline QA end-to-end
├── results/
│   ├── metrics.csv             # Kết quả benchmark chi tiết
│   ├── outputs.json            # Câu trả lời đầy đủ
│   ├── summary.csv             # Bảng tổng hợp theo method
│   └── plots/                  # Biểu đồ
│       ├── token_reduction.png
│       ├── latency.png
│       ├── similarity.png
│       ├── summary_table.png
│       └── summary_bars.png
└── tests/
```

### Bảng module

| Module | Chức năng | Input | Output |
|--------|-----------|-------|--------|
| `pdf_loader.py` | Trích xuất text từ PDF | PDF file | Raw text |
| `chunking.py` | Chia text thành chunks | Raw text | List[chunk] |
| `retrieval.py` | Tìm chunks liên quan | Query + index | Top-k chunks |
| `compressor.py` | Nén context | Query + chunks | Compressed text |
| `generator.py` | Sinh câu trả lời | Prompt | Answer text |
| `evaluation.py` | Tính metrics | Answers | Scores |
| `benchmark.py` | Chạy thí nghiệm | Config | CSV + JSON |
| `visualization.py` | Vẽ biểu đồ | CSV | PNG plots |

## 5. Pipeline hoạt động

```
┌─────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐
│  PDF    │───▶│ Text Extract │───▶│  Chunking │───▶│  Embedding   │
│ Papers  │    │  (PyMuPDF)   │    │ (500 word │    │    Index     │
└─────────┘    └──────────────┘    │  overlap) │    └──────┬───────┘
                                   └───────────┘           │
                                                           ▼
┌─────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────────┐
│ Answer  │◀───│  LLM Gen     │◀───│  Context  │◀───│  Semantic    │
│         │    │  (Ollama)    │    │Compression│    │  Retrieval   │
└────┬────┘    └──────────────┘    └───────────┘    │  (top-k)     │
     │                                              └──────────────┘
     ▼
┌──────────────┐    ┌───────────┐
│  Evaluation  │───▶│  Results  │
│  (metrics)   │    │ CSV/JSON  │
└──────────────┘    └───────────┘
```

### Chi tiết từng bước

1. **PDF Ingestion:** Dùng PyMuPDF (fitz) trích xuất toàn bộ text từ PDF
2. **Chunking:** Chia text thành chunks 500 từ, overlap 50 từ
3. **Indexing:** Encode tất cả chunks bằng sentence-transformers → vector embeddings
4. **Retrieval:** Khi có câu hỏi, encode query → cosine similarity → lấy top-k chunks
5. **Compression:** Áp dụng phương pháp compression để giảm context
6. **Generation:** Đưa compressed context + query vào LLM để sinh câu trả lời
7. **Evaluation:** So sánh answer từ compressed vs full context

## 6. Các phương pháp Compression

| Phương pháp | Mô tả | Ưu điểm | Nhược điểm |
|-------------|--------|----------|------------|
| **Full** | Không nén, dùng toàn bộ chunks (baseline) | Không mất thông tin | Token nhiều, tốn chi phí |
| **Extractive** | Chọn top-k câu theo semantic similarity với query | Nhanh, giữ nguyên wording | Có thể chọn câu trùng lặp |
| **MMR** | Maximal Marginal Relevance — cân bằng relevance và diversity | Giảm trùng lặp | Chậm hơn extractive |
| **Summary** | Dùng LLM tóm tắt context theo hướng query | Nén tỉ lệ cao nhất | Thêm latency, có thể mất chi tiết |
| **Hybrid** | Extractive scoring + MMR diversity filtering | Cân bằng tốt nhất | Phức tạp hơn |

### Chi tiết thuật toán

**Extractive Compression:**
```
1. Tách mỗi chunk thành sentences
2. Encode query và tất cả sentences
3. Tính cosine similarity(query, sentence) + 0.2 × retrieval_score
4. Sắp xếp theo score giảm dần
5. Chọn top-k sentences (loại trùng lặp)
```

**MMR Compression:**
```
1. Tách và encode tất cả sentences
2. Tính query_sim và sentence-sentence similarity
3. Greedy selection:
   MMR_score = λ × relevance(q, s) − (1−λ) × max_redundancy(s, selected)
4. Chọn sentence có MMR_score cao nhất, lặp lại
```

**Hybrid Compression:**
```
1. Bước Extractive: Score tất cả sentences theo query similarity
2. Pre-filter: Giữ top 2×k candidates
3. Bước MMR: Áp dụng MMR trên candidates để loại trùng lặp
4. Output: Top-k diverse, relevant sentences
```

## 7. Evaluation Metrics

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **Token Reduction %** | `((before − after) / before) × 100` | Tỉ lệ giảm token sau compression |
| **Semantic Similarity** | `cosine_sim(emb(full_answer), emb(compressed_answer))` | Độ tương đồng ngữ nghĩa giữa hai câu trả lời |
| **ROUGE-L F1** | LCS-based F1 giữa reference và hypothesis | Overlap từ vựng giữa hai câu trả lời |
| **Latency** | Đo thời gian retrieval, compression, generation | Hiệu suất pipeline |

### Giải thích metrics

- **Token Reduction:** Đo lượng token tiết kiệm được. Reduction cao = hiệu quả compression cao.
- **Semantic Similarity:** So sánh answer từ full context vs compressed context. Score gần 1.0 = compression không làm mất chất lượng.
- **ROUGE-L:** Đo overlap từ vựng. Bổ sung cho semantic similarity.
- **Baseline comparison:** Method "full" luôn có reduction = 0%, similarity = 1.0. Các method khác được so sánh với baseline này.

## 8. Benchmark Experiments

### Dataset

| Paper | Nguồn | Chủ đề |
|-------|-------|--------|
| Attention Is All You Need | arXiv:1706.03762 | Transformer architecture |
| RAG | arXiv:2005.11401 | Retrieval-Augmented Generation |
| Longformer | arXiv:2004.05150 | Long-document attention |
| LLMLingua | arXiv:2310.05736 | Prompt compression |
| Lost in the Middle | arXiv:2307.03172 | Position bias in LLM |

Mỗi paper có **5 câu hỏi** đánh giá, tổng cộng **25 câu hỏi × 5 methods = 125 experiments**.

### Kết quả mong đợi

| Method | Avg Token Reduction | Avg Similarity | Avg Latency |
|--------|:---:|:---:|:---:|
| Full (baseline) | 0.0% | 1.000 | Baseline |
| Extractive | ~60-70% | ~0.85-0.95 | Thấp |
| MMR | ~60-70% | ~0.85-0.95 | Thấp |
| Summary | ~70-85% | ~0.75-0.90 | Cao (thêm LLM call) |
| Hybrid | ~60-70% | ~0.85-0.95 | Trung bình |

### Ví dụ output (metrics.csv)

```csv
paper,question,method,tokens_before,tokens_after,reduction_percent,latency_total,similarity_score,rouge_l_f1
attention_is_all_you_need.pdf,What is the main contribution?,full,2500,2500,0.0,3.26,1.000,1.000
attention_is_all_you_need.pdf,What is the main contribution?,extractive,2500,820,67.2,2.85,0.921,0.784
attention_is_all_you_need.pdf,What is the main contribution?,hybrid,2500,850,66.0,2.57,0.934,0.812
```

## 9. Hướng dẫn cài đặt

### Yêu cầu

- Python >= 3.10
- [Ollama](https://ollama.ai) (cho LLM inference local)
- GPU recommended (cho sentence-transformers)

### Bước 1: Clone và cài dependencies

```bash
git clone <repository-url>
cd CK

pip install -r requirements.txt
```

### Bước 2: Cài đặt và chạy Ollama

```bash
# Cài Ollama (xem https://ollama.ai)
ollama pull qwen2.5:7b-instruct
ollama serve
```

### Bước 3: Chuẩn bị dataset

Tải PDF papers từ arXiv và đặt vào `data/papers/`:

| Paper | Link tải | Tên file |
|-------|----------|----------|
| Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | `Attention.pdf` |
| RAG | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | `RAG.pdf` |
| Longformer | [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) | `Longformer.pdf` |
| LLMLingua | [arXiv:2310.05736](https://arxiv.org/abs/2310.05736) | `LLMLingua.pdf` |
| Lost in the Middle | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) | `LostintheMiddle.pdf` |

> **Lưu ý:** Tên file phải khớp với key trong `data/questions.json`.

## 10. Hướng dẫn chạy Benchmark

### Chạy full benchmark

```bash
python main.py --benchmark
```

Pipeline sẽ tự động:
1. Load tất cả PDF từ `data/papers/`
2. Chunking và build embedding index
3. Chạy tất cả câu hỏi × tất cả methods
4. Lưu kết quả vào `results/`
5. Generate biểu đồ vào `results/plots/`

### Xem bảng tổng hợp

```bash
python main.py --summary
```

### Chế độ interactive (hỏi đáp)

```bash
python main.py
```

Trong interactive mode:
- Gõ câu hỏi để nhận câu trả lời
- `method extractive` — đổi phương pháp compression
- `exit` — thoát

### Generate lại biểu đồ

```bash
python main.py --plot
```

### Tuỳ chỉnh đường dẫn

```bash
python main.py --benchmark \
    --papers-dir path/to/papers \
    --questions path/to/questions.json \
    --results-dir path/to/results
```

## 11. Ví dụ kết quả

### Bảng tổng hợp benchmark (summary.csv)

| Method | Tokens Before | Tokens After | Reduction % | Latency (s) | Similarity | ROUGE-L F1 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Full | 2500 | 2500 | 0.0% | 3.26 | 1.000 | 1.000 |
| Extractive | 2500 | 825 | 67.0% | 2.85 | 0.912 | 0.756 |
| MMR | 2500 | 840 | 66.4% | 2.90 | 0.905 | 0.742 |
| Summary | 2500 | 420 | 83.2% | 5.15 | 0.845 | 0.632 |
| Hybrid | 2500 | 815 | 67.4% | 2.95 | 0.928 | 0.781 |

> *Lưu ý: Số liệu trên là ví dụ minh hoạ. Kết quả thực tế phụ thuộc vào papers, model, và hardware.*

### Output files

Sau khi chạy benchmark, thư mục `results/` chứa:

- **`metrics.csv`** — Metrics chi tiết cho từng experiment (paper × question × method)
- **`outputs.json`** — Câu trả lời đầy đủ, context, và metadata
- **`summary.csv`** — Bảng tổng hợp average metrics theo method
- **`plots/`** — 5 biểu đồ phân tích

## 12. Visualization

Benchmark tự động generate 5 loại biểu đồ:

| Biểu đồ | File | Nội dung |
|----------|------|----------|
| Token Reduction | `plots/token_reduction.png` | So sánh tokens before/after theo method |
| Latency | `plots/latency.png` | Phân tích latency: retrieval, compression, generation |
| Similarity | `plots/similarity.png` | Semantic similarity và ROUGE-L F1 theo method |
| Summary Table | `plots/summary_table.png` | Bảng tổng hợp dạng hình ảnh |
| Summary Bars | `plots/summary_bars.png` | 3 biểu đồ cột: reduction, similarity, latency |

## 13. Thêm paper mới

1. Đặt file PDF vào `data/papers/`
2. Thêm câu hỏi vào `data/questions.json`:

```json
{
    "ten_paper.pdf": [
        "Câu hỏi 1?",
        "Câu hỏi 2?",
        "Câu hỏi 3?"
    ]
}
```

3. Chạy benchmark: `python main.py --benchmark`

## 14. Technology Stack

| Thành phần | Công nghệ |
|-----------|-----------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF Processing | PyMuPDF (fitz) |
| LLM | Ollama (`qwen2.5:7b-instruct`) |
| Visualization | matplotlib |
| Evaluation | Custom (cosine similarity, ROUGE-L, token counting) |

## 15. Limitations

- **Approximate token counting:** Đếm token bằng word split, không dùng tokenizer chính xác của LLM
- **Dataset nhỏ:** 6 papers, 30 câu hỏi — chưa đủ lớn cho kết luận thống kê mạnh
- **Semantic similarity là proxy metric:** Cosine similarity giữa embeddings không hoàn toàn phản ánh chất lượng thực sự
- **Không có human evaluation:** Chưa có đánh giá từ người dùng thực tế
- **Single LLM:** Chỉ test với một model (qwen2.5:7b-instruct)
- **PDF extraction noise:** Một số PDF có bảng/hình ảnh khó extract text chính xác

## 16. Future Work

- **Adaptive compression:** Tự động chọn phương pháp compression phù hợp theo loại câu hỏi
- **Hierarchical summarization:** Tóm tắt theo cấp bậc cho tài liệu rất dài
- **Query-aware chunking:** Chia chunks dựa trên ngữ nghĩa thay vì word count cố định
- **Larger benchmark:** Mở rộng dataset với nhiều papers và câu hỏi hơn
- **Multi-model comparison:** Test với nhiều LLM khác nhau (Llama, Mistral, Gemma)
- **Human evaluation:** Đánh giá chất lượng bởi người dùng thực tế
- **Token-level compression:** Implement phương pháp compression ở cấp token (như LLMLingua)
- **Cross-lingual support:** Hỗ trợ tài liệu tiếng Việt

## License

Đồ án học thuật — chỉ phục vụ mục đích giáo dục và nghiên cứu.
