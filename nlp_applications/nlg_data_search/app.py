"""
Web server for hybrid recipe search (Assignment 1, Group 108).

To run: uvicorn app:app --reload
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from search import search_with_context

app = FastAPI(title="Hybrid Recipe Search — Group 108")


class Query(BaseModel):
    query: str
    top_k: int = 10


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/search")
def search(req: Query):
    # Run BM25 and dense search, combine results using RRF, and return the top matches.
    return {"query": req.query, "results": search_with_context(req.query, req.top_k)}
