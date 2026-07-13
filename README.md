# ML Research

A collection of course projects and assignments exploring machine learning, natural language generation, computer vision, and conversational AI.

---

## Projects

### 1. Domain LLM Adaptation & Production Optimization — QLoRA Fine-Tuning
**Path:** [`llmforgenai/chatbot-peft/`](llmforgenai/chatbot-peft/)

Adapts **GPT-2-Large (774M)** to the scientific research papers domain using QLoRA, then optimizes for production:
- **Part A** — Domain data collection from arXiv PDFs, baseline model evaluation
- **Part B** — Instruction dataset creation & QLoRA fine-tuning (LoRA rank=16, alpha=32)
- **Part C** — Inference optimization: 5 decoding strategies, speculative decoding (GPT-2 124M as draft model), 4-bit quantization & production cost analysis

**Tech:** GPT-2, PEFT/LoRA, bitsandbytes, speculative decoding, HuggingFace Transformers

### 2. Controlled Natural Language Generation
**Path:** [`conversational_ai/`](conversational_ai/)

Compares a **custom Decoder-only Transformer** (trained from scratch) against **Qwen 2.5 3B Instruct** for slot-conditioned response generation in an e-commerce order assistant domain:
- **Module 1** — Intent-slot schema design, input linearization & tokenization
- **Module 2** — Custom decoder architecture (2 layers, 4 attention heads), LLM prompting
- **Module 3** — Quantitative evaluation (Slot Accuracy, BLEU, ROUGE-L) and qualitative analysis with deployment recommendation

**Tech:** PyTorch (custom Transformer), HuggingFace Transformers (Qwen 2.5 3B)

### 3. Hybrid Search QA Optimization — BM25 + Dense Retrieval
**Path:** [`nlp_applications/nlg_data_search/`](nlp_applications/nlg_data_search/)

**NLP Applications — Assignment 1, Group 108: Domain 3 — Culinary Recipe Databases.**

A hybrid search system combining **BM25 keyword retrieval** with **MiniLM dense embeddings**, fused via **Reciprocal Rank Fusion (RRF)** over 1,500 RecipeNLG passages:

| File | Task |
|---|---|
| `prepare_data.py` | Build 1,500 passages from RecipeNLG |
| `build_index.py` | Build BM25 + dense indexes |
| `search.py` | BM25, dense, hybrid (RRF) — Tasks 1, 2, 3 |
| `evaluate.py` | MAP & MRR evaluation |
| `app.py` | FastAPI backend with web UI |

**Tech:** FastAPI, sentence-transformers (all-MiniLM-L6-v2), BM25 (rank-bm25), RRF

### 4. 3D Computer Vision — Projections & Feature Matching
**Path:** [`computervision/projections/`](computervision/projections/)

**Computer Vision — Assignment 1** covering:
- **Task 1** — Perspective vs. orthographic projection (visual comparison)
- **Task 2** — Rigid-body transform functions (rotation, translation)
- **Task 3** — SIFT feature matching and affine transformation estimation

**Tech:** OpenCV, NumPy, Matplotlib

---

## Getting Started

Projects use different dependencies. Each directory has its own requirements or setup. For specific instructions, see the project-level `README.md` (notably [`nlp_applications/nlg_data_search/README.md`](nlp_applications/nlg_data_search/README.md)).

The notebooks are designed to run on **Google Colab** (T4 GPU) and include all setup/install cells.
