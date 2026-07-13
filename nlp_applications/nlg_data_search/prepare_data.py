"""
Sample 1,500 culinary recipes from the RecipeNLG CSV and save them as passages.
"""
import json
import os
import re

import pandas as pd

CSV_PATH = "archive/RecipeNLG_dataset.csv"
OUT_PATH = "data/passages.jsonl"


def parse_list(s):
    # The RecipeNLG dataset stores ingredients and directions as JSON strings inside each cell.
    if not isinstance(s, str):
        return []
    try:
        return [str(x).strip() for x in json.loads(s) if str(x).strip()]
    except Exception:
        return []


def squish(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Put RecipeNLG csv at {CSV_PATH}")
    os.makedirs("data", exist_ok=True)

    # Read only the first 50,000 rows to avoid loading the full 2.3 GB file into memory.
    df = pd.read_csv(CSV_PATH, nrows=50_000,
                     usecols=["title", "ingredients", "directions", "NER", "link"])
    df = df.dropna(subset=["title", "ingredients", "directions"])
    df = df.drop_duplicates(subset="title")
    df = df.sample(n=1500, random_state=42).reset_index(drop=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            ingredients = parse_list(row["ingredients"])
            directions = parse_list(row["directions"])
            if not ingredients or not directions:
                continue
            title = squish(str(row["title"]))
            text = squish(f"{title}. Ingredients: {'; '.join(ingredients)}. "
                          f"Directions: {' '.join(directions)}")[:2000]
            f.write(json.dumps({
                "id": f"recipe-{i:05d}",
                "title": title,
                "text": text,
                "link": str(row.get("link", "")),
                "ner": parse_list(row.get("NER", "[]")),
            }) + "\n")

    print(f"Wrote 1500 passages to {OUT_PATH}")


if __name__ == "__main__":
    main()
