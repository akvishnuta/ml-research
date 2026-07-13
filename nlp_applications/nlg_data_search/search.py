"""
Implements BM25 (keyword-based) and MiniLM (semantic) search, combined
using Reciprocal Rank Fusion to produce hybrid search results.

Build indexes first with:  python build_index.py
"""
import json
import os
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PASSAGES = "data/passages.jsonl"
BM25_FILE = "artifacts/bm25.pkl"
EMB_FILE = "artifacts/embeddings.npy"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # smoothing constant from the original RRF paper (Cormack et al. 2009)


# --- tokenizer and corpus loading -------------------------------------------------------

# Remove common unit words so they don't inflate scores (e.g. "tsp", "oz", "cup").
STOPWORDS = set("a an and are as at be by for from has have in into is it its "
                "of on or that the this to was were will with you c tsp tbsp "
                "oz lb lbs pkg can cans cup cups about until minutes hours".split())

def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]

_corpus = None
def load_corpus():
    global _corpus
    if _corpus is None:
        with open(PASSAGES, encoding="utf-8") as f:
            _corpus = [json.loads(line) for line in f if line.strip()]
    return _corpus


# --- BM25 keyword search ---------------------------------------------------------------------

_bm25 = None
def get_bm25():
    global _bm25
    if _bm25 is None:
        with open(BM25_FILE, "rb") as f:
            _bm25 = pickle.load(f)
    return _bm25

def build_bm25():
    os.makedirs("artifacts", exist_ok=True)
    tokens = [tokenize(d["title"] + " " + d["text"]) for d in load_corpus()]
    bm25 = BM25Okapi(tokens)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 indexed {len(tokens)} docs")

def bm25_search(query, k):
    scores = get_bm25().get_scores(tokenize(query))
    top = np.argpartition(scores, -k)[-k:]
    top = top[np.argsort(scores[top])[::-1]]
    return [(int(i), float(scores[i])) for i in top]


# --- Dense semantic search using MiniLM -----------------------------------------------------------

_model = None
_emb = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(DENSE_MODEL)
    return _model

def get_emb():
    global _emb
    if _emb is None:
        _emb = np.load(EMB_FILE)
    return _emb

def build_embeddings():
    os.makedirs("artifacts", exist_ok=True)
    corpus = load_corpus()
    texts = [d["title"] + ". " + d["text"] for d in corpus]
    emb = get_model().encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)
    np.save(EMB_FILE, emb)
    print(f"Dense embeddings: {emb.shape}")

def dense_search(query, k):
    q = get_model().encode([query], normalize_embeddings=True)[0].astype(np.float32)
    sims = get_emb() @ q  # embeddings are normalized, so dot product equals cosine similarity
    top = np.argpartition(sims, -k)[-k:]
    top = top[np.argsort(sims[top])[::-1]]
    return [(int(i), float(sims[i])) for i in top]


# --- Hybrid search using Reciprocal Rank Fusion ---------------------------------------

def hybrid_search(query, k, fusion_depth=100):
    """Merge BM25 and dense results using RRF over the top `fusion_depth` hits from each."""
    sparse_hits = bm25_search(query, fusion_depth)
    dense_hits = dense_search(query, fusion_depth)
    fused = {}
    for ranked in (sparse_hits, dense_hits):
        for rank, (idx, _) in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return top


def search_with_context(query, k=10):
    """Run hybrid search and return the top results with title, score, and text for the API."""
    corpus = load_corpus()
    out = []
    for rank, (idx, score) in enumerate(hybrid_search(query, k), start=1):
        doc = corpus[idx]
        out.append({
            "rank": rank,
            "title": doc["title"],
            "score": round(score, 6),
            "context": doc["text"],
            "link": doc.get("link", ""),
        })
    return out
