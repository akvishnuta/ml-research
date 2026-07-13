"""
Build BM25 and dense (embedding) indexes.
"""
from search import build_bm25, build_embeddings

if __name__ == "__main__":
    build_bm25()
    build_embeddings()
