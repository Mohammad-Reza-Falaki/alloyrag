# src/helpers/analyze_general_tables.py
"""
Analyzes what content general-type tables actually contain.
Run: python src/helpers/analyze_general_tables.py
"""
import sys
from pathlib import Path

# Add src/ to path so helpers can import from main modules
sys.path.insert(0, str(Path(__file__).parent.parent))


import json
import re
from pathlib import Path
from collections import defaultdict
from extract_papers import normalize_text, TABLE_TYPE_KEYWORDS

CORPUS_DIR = Path("data/corpus")

def analyze_general_tables():
    general_tables    = []
    keyword_hit_map   = defaultdict(int)

    for path in CORPUS_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue

        for table in doc.get("property_tables", []):
            if table.get("table_type") != "general":
                continue

            # Build searchable text
            header    = table.get("header", [])
            rows      = table.get("rows", [])
            caption   = table.get("caption", "")
            raw_text  = " ".join(str(h) for h in header)
            raw_text += " " + caption
            for row in rows[:3]:
                raw_text += " " + " ".join(str(v) for v in row.values())
            search_text = normalize_text(raw_text)

            # Count keyword hits per type
            type_hits = {}
            for ttype, config in TABLE_TYPE_KEYWORDS.items():
                hits = sum(
                    1 for kw in config["keywords"]
                    if kw in search_text
                )
                type_hits[ttype] = hits

            # What is the best single type score?
            best_type  = max(type_hits, key=type_hits.get)
            best_score = type_hits[best_type]

            general_tables.append({
                "paper_id":  doc["paper_id"],
                "caption":   caption[:80],
                "header":    header[:5],
                "type_hits": type_hits,
                "best_type": best_type,
                "best_score": best_score,
            })

            keyword_hit_map[f"{best_type}={best_score}"] += 1

    print(f"Total general tables: {len(general_tables)}\n")

    # Score distribution
    print("Score distribution (best_type=score : count)")
    print("=" * 50)
    for key, count in sorted(
        keyword_hit_map.items(),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {key:<35} {count}")

    # Tables with score=1 (just missed threshold)
    missed = [t for t in general_tables if t["best_score"] == 1]
    print(f"\nTables with score=1 (missed by 1 match): {len(missed)}")
    print("Sample headers from these tables:")
    for t in missed[:10]:
        print(f"  [{t['best_type']}] {t['header']}")
        print(f"    Caption: {t['caption'][:60]}")

    # Tables with score=0 (no matches at all)
    zero = [t for t in general_tables if t["best_score"] == 0]
    print(f"\nTables with score=0 (no keyword matches): {len(zero)}")
    print("Sample headers:")
    for t in zero[:10]:
        print(f"  {t['header']}")
        print(f"  Caption: {t['caption'][:60]}")

if __name__ == "__main__":
    analyze_general_tables()