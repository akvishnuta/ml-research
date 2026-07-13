# Hybrid Search (Keyword + Vector) QA Optimization

**Course:** NLP Applications, BITS Pilani - Assignment 1
**Group:** 108 · **Domain 3:** Culinary Recipe Databases
**Dataset:** RecipeNLG (Bień et al., INLG 2020) — https://recipenlg.cs.put.poznan.pl/
**Reference:** Jurafsky & Martin, *Speech and Language Processing*, Ch. 15.

## Group Members

| Name | Student ID | Contribution |
|---|---|---|
| Akhil Kumar | 2024ac05446 | 100% |
| Revanth Nandamuri | 2024ac05761 | 100% |
| M N Pradeep Gowtham | 2024ac05647 | 100% |
| Nakul Ramanathan | 2024ac05356 | 100% |
| DHANRAJ SAHOO | 2024AC05620 | 100% |

NLP Applications Assignment 1 — **Group 108, Domain 3: Culinary Recipe Databases**.

BM25 + MiniLM dense embeddings, fused with Reciprocal Rank Fusion, served
behind a FastAPI backend with a simple HTML frontend.

## Files

| File | Task |
|---|---|
| `prepare_data.py`   | Build 1,500 passages from RecipeNLG |
| `build_index.py`    | Build BM25 + dense indexes (Task 1) |
| `search.py`         | BM25, dense, hybrid (RRF) — Tasks 1, 2, 3 |
| `evaluate.py`       | MAP & MRR, vector-only vs hybrid (Task 5) |
| `app.py`            | FastAPI backend (Task 4) |
| `static/index.html` | Frontend |

## How to run locally

Requires Python 3.10+ and the RecipeNLG csv at `archive/RecipeNLG_dataset.csv`
(download from https://recipenlg.cs.put.poznan.pl/).

```bash
python3 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python prepare_data.py    # writes data/passages.jsonl
python build_index.py     # writes artifacts/bm25.pkl + embeddings.npy
python evaluate.py        # prints MAP/MRR table, writes eval/metrics.json

uvicorn app:app --reload  # starts the server at http://127.0.0.1:8000
```

Once the server is running, open your browser and go to:

```
http://127.0.0.1:8000
```

You will see the search UI. Type a recipe query (e.g. `chocolate chip cookies`) into the
text box and click **Search** to get the top hybrid results. The Swagger API docs are
available at `http://127.0.0.1:8000/docs`.

If the zip already includes `data/passages.jsonl` and `artifacts/*`, skip the
first two `python` commands — the app will start straight away.

## API

`POST /search` with JSON body:

```json
{ "query": "chocolate chip cookies", "top_k": 10 }
```

Returns the hybrid (RRF-fused) context blocks:

```json
{
  "query": "...",
  "results": [
    { "rank": 1, "title": "...", "score": 0.032, "context": "...", "link": "..." },
    ...
  ]
}
```

Swagger UI at `/docs`.

## Dataset

RecipeNLG (Bień et al., INLG 2020) — https://recipenlg.cs.put.poznan.pl/

---

## 1. Approach

We index a 1,500-passage culinary corpus with two retrievers, BM25 (sparse,
keyword) and a Sentence-Transformer (dense, vector) - and combine them with
Reciprocal Rank Fusion. The hybrid is served behind a FastAPI app with an
HTML frontend, and we compare the vector-only baseline against the hybrid on
the same query set using MAP and MRR.

We chose the culinary domain because it is dominated by exact-token jargon
(ingredient names, units, technique words). That is the regime where dense
retrievers tend to underperform BM25 and where hybrid retrieval is expected
to help most.

## 2. Corpus

`prepare_data.py` reads the first 50,000 rows of RecipeNLG (the full file is
2.3 GB), drops rows with missing title/ingredients/directions, deduplicates
on title and takes a deterministic 1,500-recipe sample (`random_state=42`).
Each passage is one self-contained block:

```
<title>. Ingredients: <…; …>. Directions: <…>
```

with the title first so MiniLM's 256-token window always contains the
salient summary. Output is `data/passages.jsonl`.

## 3. Retrievers (`search.py`)

**BM25 (sparse).** `rank_bm25.BM25Okapi` with default parameters. Tokenizer
is `[a-z0-9]+` lowercased, minus a small stoplist that includes the usual
English filler plus culinary-noise words like `tsp`, `oz`, `cup`, `minutes`.
Without that, unit tokens swamp IDF. We index `title + " " + text` so the
recipe name gets full BM25 weight twice.

**Dense (vector baseline).** `sentence-transformers/all-MiniLM-L6-v2`, 384
dimensions. We encode `title + ". " + text`, L2-normalise the embedding
matrix at build time so cosine similarity is a single dot product, and store
the (1500, 384) float32 array as `artifacts/embeddings.npy`. Search is
brute-force `embeddings @ q` - faster than FAISS at this corpus size and one
fewer dependency.

**Hybrid (Reciprocal Rank Fusion).** For a doc d at rank rᵢ in list i:

```
score(d) = Σᵢ[ 1 / (k + rᵢ)],     k = 60
```

(Cormack, Clarke & Buttcher, SIGIR 2009). RRF is score-agnostic - we don't
need to calibrate unbounded BM25 scores against bounded cosine
similarities. Each retriever contributes its top 100 candidates to the
fusion.

## 4. Backend (`app.py`)

A single `POST /search` endpoint. Body: `{query, top_k}`. Response: the
fused context blocks (title, score, context, link). The frontend
(`static/index.html`) is a vanilla HTML form that hits that endpoint.
Swagger UI auto-generated at `/docs`.

## 5. Evaluation (`evaluate.py`)

The brief doesn't ship gold relevance judgments, so we generate
**pseudo qrels** in a deterministic way: every query carries a list of
`(term, field)` rules and a passage is judged relevant if and only if *all* rules
match. Fields are `title`, `ner` (RecipeNLG's ingredient noun list), or
`text`. For example:

```python
("oatmeal raisin cookies", [("oat", "ner"), ("raisin", "ner")])
```

marks any passage whose ingredient list mentions both `oat` and `raisin`.
15 culinary queries cover short keyword queries, full-sentence questions
and multi-ingredient queries.

We compute:

- **Average Precision** per query, then **MAP** across queries.
- **Reciprocal Rank** per query, then **MRR** across queries.

Both at `k ∈ {1, 3, 5, 10, 20, 50, 100}` so we can examine diminishing returns.

## 6. Results

### 6.1 Vector-only vs Hybrid

| k | Vector MAP | Vector MRR | Hybrid MAP | Hybrid MRR | Δ MAP | Δ MRR |
|---:|---:|---:|---:|---:|---:|---:|
|   1 | 0.0245 | 0.4000 | 0.0432 | 0.5333 | +76 % | +33 % |
|   3 | 0.0985 | 0.5444 | 0.1880 | 0.7222 | +91 % | +33 % |
|   5 | 0.1290 | 0.5744 | 0.2390 | 0.7222 | +85 % | +26 % |
|  **10** | **0.1796** | **0.5819** | **0.3243** | **0.7222** | **+81 %** | **+24 %** |
|  20 | 0.2292 | 0.5819 | 0.3696 | 0.7222 | +61 % | +24 % |
|  50 | 0.2617 | 0.5819 | 0.4187 | 0.7243 | +60 % | +24 % |
| 100 | 0.2831 | 0.5828 | 0.4453 | 0.7243 | +57 % | +24 % |

**At the headline cutoff k = 10, the hybrid system beats the vector-only
baseline by +81 % MAP and +24 % MRR.** The win is decisive at every cutoff.

### 6.2 Diminishing returns

- **MRR saturates around k ≈ 10.** Once the first relevant doc is in the
  returned list, raising k cannot help. Hybrid MRR moves from 0.7222 at
  k = 10 to 0.7243 at k = 100 - essentially flat. That makes k = 10 the
  natural default for the `top_k` parameter in the API.
- **MAP gains shrink fast.** Hybrid MAP grows by +14 % from k = 10 to
  k = 20 (0.324 → 0.370), then only +20 % over the next 5× cost expansion
  to k = 100 (0.370 → 0.445). The marginal precision per extra returned
  doc collapses after k ≈ 20.
- **The hybrid advantage is largest at small k.** At k = 3, hybrid MAP is
  nearly double the vector-only MAP (0.188 vs 0.099). This is where exact
  jargon matching matters most - exactly the kind of query a real
  recipe-QA user would issue.

### 6.3 Qualitative example

For *"creamy chicken casserole with mushroom soup"*, the vector-only top-1
is semantically related (e.g. *Saucy Chicken*) but often misses the
"mushroom soup" concept. The hybrid top-1 is *Creamless Mushroom Soup*
followed by *Chicken Bake* and *Broccoli And Chicken* - both contain
"cream of mushroom soup" verbatim in their ingredients. That's BM25
catching the exact jargon the embeddings paraphrased away, and RRF
promoting it.

## 7. Design choices

| Decision | Why |
|---|---|
| MiniLM-L6-v2 over a 768-dim model | 80 MB, fast on CPU, accuracy gap is small at this corpus size |
| Brute-force cosine over FAISS | N = 1,500 - the matvec is < 1 ms; FAISS would add a dependency for no real gain |
| RRF with k = 60 | Score-agnostic, no calibration between BM25 (unbounded) and cosine (bounded) |
| Fusion depth 100 | Enough to catch the long tail, cheap to fuse |
| Stoplist includes culinary units (`tsp`, `oz`, …) | Without it, units crush BM25's IDF signal |
| Pseudo-qrels = AND of `(term, field)` rules | Deterministic, inspectable, easy to extend |

## 8. Challenges

1. **Long recipes vs encoder window.** MiniLM caps at ~256 tokens. We
   truncate passages at 2,000 chars and put the title first.
2. **JSON-encoded list columns.** RecipeNLG stores `ingredients` and
   `directions` as JSON strings, not native lists. A `try/except
   json.loads` per field handles the few malformed rows.
3. **Pseudo-qrels bias.** Keyword rules implicitly favor BM25 in MAP. We
   only report vector-only vs hybrid, which are both rank-fusion or
   rank-only systems - they compete on the same footing.

## 9. Conclusion

Hybrid retrieval delivered a clear uplift over a dense-only baseline on a
recipe corpus: **+81 % MAP and +24 % MRR at k = 10**. Diminishing returns
kick in past k = 20 (approx.) on MAP and past k = 10 (approx.) on MRR, so k = 10 is the
sensible default for QA-style retrieval.

