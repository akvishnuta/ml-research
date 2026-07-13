"""
Compare vector-only search vs hybrid search using MAP and MRR metrics,
and sweep over different values of top-k to see how performance changes.

Relevance labels are generated automatically: a recipe is considered relevant
to a query if all the specified terms appear in the right fields (title, NER tags,
or full text).
"""
import json
import os

import numpy as np

from search import dense_search, hybrid_search, load_corpus

OUT_FILE = "eval/metrics.json"

# 15 culinary queries covering short phrases, full sentences, and multi-ingredient searches.
QUERIES = [
    ("creamy chicken casserole with mushroom soup", [("chicken", "ner"), ("mushroom", "ner")]),
    ("vegetarian broccoli cheese bake",             [("broccoli", "ner"), ("cheese", "ner")]),
    ("how to bake chocolate chip cookies",          [("chocolate chip", "text"), ("cookie", "title")]),
    ("beef stew with potatoes and carrots",         [("beef", "ner"), ("potato", "ner")]),
    ("strawberry shortcake dessert",                [("strawberry", "ner"), ("cake", "title")]),
    ("creamy pumpkin pie filling",                  [("pumpkin", "ner"), ("pie", "title")]),
    ("homemade pizza dough yeast",                  [("yeast", "ner"), ("dough", "text")]),
    ("lemon cake with frosting",                    [("lemon", "ner"), ("cake", "title")]),
    ("slow cooker pulled pork",                     [("pork", "ner"), ("slow", "text")]),
    ("banana bread with walnuts",                   [("banana", "ner"), ("nut", "ner")]),
    ("apple pie cinnamon crust",                    [("apple", "ner"), ("cinnamon", "ner")]),
    ("homemade salsa with tomatoes and onions",     [("tomato", "ner"), ("onion", "ner")]),
    ("fried rice with soy sauce and vegetables",    [("rice", "ner"), ("soy sauce", "text")]),
    ("shrimp pasta with garlic and parmesan",       [("shrimp", "ner"), ("garlic", "ner")]),
    ("oatmeal raisin cookies",                      [("oat", "ner"), ("raisin", "ner")]),
]


def field_of(doc, field):
    if field == "title": return doc["title"].lower()
    if field == "ner":   return " ".join(doc.get("ner", [])).lower()
    return doc["text"].lower()


def relevant_for(query_rules, corpus):
    rels = set()
    for i, doc in enumerate(corpus):
        if all(term in field_of(doc, fld) for term, fld in query_rules):
            rels.add(i)
    return rels


def average_precision(ranked, rels):
    if not rels:
        return 0.0
    hits, prec_sum = 0, 0.0
    for rank, idx in enumerate(ranked, 1):
        if idx in rels:
            hits += 1
            prec_sum += hits / rank
    return prec_sum / len(rels) if hits else 0.0


def reciprocal_rank(ranked, rels):
    for rank, idx in enumerate(ranked, 1):
        if idx in rels:
            return 1.0 / rank
    return 0.0


def main():
    os.makedirs("eval", exist_ok=True)
    corpus = load_corpus()

    # Build the relevance set once per query so we do not repeat the work.
    qrels = [(q, rules, relevant_for(rules, corpus)) for q, rules in QUERIES]
    print(f"{sum(1 for _,_,r in qrels if r)} queries have at least one relevant doc")

    methods = {"vector-only": dense_search, "hybrid": hybrid_search}
    k_values = [1, 3, 5, 10, 20, 50, 100]

    results = {}
    for name, fn in methods.items():
        results[name] = {}
        for k in k_values:
            aps, rrs = [], []
            for query, _rules, rels in qrels:
                if not rels:
                    continue
                ranked = [i for i, _ in fn(query, k)]
                aps.append(average_precision(ranked, rels))
                rrs.append(reciprocal_rank(ranked, rels))
            results[name][k] = {"MAP": float(np.mean(aps)), "MRR": float(np.mean(rrs))}

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Print results in a readable table.
    print(f"\nVector-only vs Hybrid (MAP / MRR):\n")
    print(f"{'k':>5s}  {'vector MAP':>11s}  {'vector MRR':>11s}  {'hybrid MAP':>11s}  {'hybrid MRR':>11s}")
    print("-" * 60)
    for k in k_values:
        v, h = results["vector-only"][k], results["hybrid"][k]
        print(f"{k:>5d}  {v['MAP']:>11.4f}  {v['MRR']:>11.4f}  {h['MAP']:>11.4f}  {h['MRR']:>11.4f}")

    print(f"\nFull JSON: {OUT_FILE}")


if __name__ == "__main__":
    main()
