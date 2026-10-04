# src/helpers/corpus_stats.py
"""
Generates exact statistics for Table 1 in the paper.
Run after extract_papers.py to get all numbers needed.
Run: python src/helpers/corpus_stats.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from collections import defaultdict

CORPUS_DIR    = Path("data/corpus")
METADATA_PATH = Path("data/raw_metadata/papers_metadata.json")
PDF_MAP_PATH  = Path("data/raw_metadata/pdf_map.json")


def get_corpus_stats():

    # Load metadata and pdf map
    with open(METADATA_PATH, encoding="utf-8") as f:
        all_metadata = json.load(f)
    with open(PDF_MAP_PATH, encoding="utf-8") as f:
        pdf_map = json.load(f)

    # Build lookup: paper_id -> alloy_system
    alloy_map = {
        m["paper_id"]: m.get("alloy_system", "Unknown")
        for m in all_metadata
    }

    # Stats containers
    stats = defaultdict(lambda: {
        "total":         0,
        "full_text":     0,
        "abstract_only": 0,
        "total_chunks":  0,
        "total_tables":  0,
        "table_types":   defaultdict(int),
        "years":         [],
    })

    corpus_files = list(CORPUS_DIR.glob("*.json"))
    print(f"Reading {len(corpus_files)} corpus files...\n")

    missing_from_corpus = 0

    for path in corpus_files:
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"  Skip {path.name}: {e}")
            continue

        pid         = doc.get("paper_id", "")
        alloy       = doc.get("alloy_system", "Unknown")
        has_full    = doc.get("has_full_text", False)
        n_chunks    = doc.get("chunk_count", 0)
        n_tables    = doc.get("table_count", 0)
        type_counts = doc.get("table_type_counts", {})
        year        = doc.get("year")

        s = stats[alloy]
        s["total"]        += 1
        s["total_chunks"] += n_chunks
        s["total_tables"] += n_tables

        if has_full:
            s["full_text"] += 1
        else:
            s["abstract_only"] += 1

        for ttype, count in type_counts.items():
            s["table_types"][ttype] += count

        if year:
            s["years"].append(year)

    # ── Table 1 ────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("TABLE 1 — CORPUS STATISTICS (copy these numbers into your paper)")
    print("=" * 78)
    print(
        f"{'Alloy':<12} {'Papers':<9} {'Full Text':<12} "
        f"{'Abstract Only':<16} {'Tables':<9} {'Chunks':<9} {'Year Range'}"
    )
    print("-" * 78)

    alloy_order = ["Al", "Ti", "Steel", "Mg", "Unknown"]
    totals = {
        "total": 0, "full_text": 0, "abstract_only": 0,
        "total_tables": 0, "total_chunks": 0,
    }
    all_years = []

    for alloy in alloy_order:
        if alloy not in stats:
            continue
        s        = stats[alloy]
        years    = sorted(s["years"])
        yr_range = f"{min(years)}–{max(years)}" if years else "N/A"

        print(
            f"{alloy:<12} {s['total']:<9} {s['full_text']:<12} "
            f"{s['abstract_only']:<16} {s['total_tables']:<9} "
            f"{s['total_chunks']:<9} {yr_range}"
        )

        totals["total"]         += s["total"]
        totals["full_text"]     += s["full_text"]
        totals["abstract_only"] += s["abstract_only"]
        totals["total_tables"]  += s["total_tables"]
        totals["total_chunks"]  += s["total_chunks"]
        all_years.extend(s["years"])

    print("-" * 78)
    yr_range = (
        f"{min(all_years)}–{max(all_years)}" if all_years else "N/A"
    )
    print(
        f"{'TOTAL':<12} {totals['total']:<9} {totals['full_text']:<12} "
        f"{totals['abstract_only']:<16} {totals['total_tables']:<9} "
        f"{totals['total_chunks']:<9} {yr_range}"
    )
    print("=" * 78)

    # ── Table type breakdown ───────────────────────────────────────────────────
    print("\nTABLE TYPE BREAKDOWN (for paper text)")
    print("=" * 45)
    all_type_counts: dict[str, int] = defaultdict(int)
    for alloy, s in stats.items():
        for ttype, count in s["table_types"].items():
            all_type_counts[ttype] += count

    for ttype, count in sorted(
        all_type_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = count / max(totals["total_tables"], 1) * 100
        print(f"  {ttype:<30} {count:<6} ({pct:.1f}%)")
    print(f"  {'TOTAL':<30} {sum(all_type_counts.values())}")
    print("=" * 45)

    # ── Table types per alloy ──────────────────────────────────────────────────
    print("\nTABLE TYPES PER ALLOY SYSTEM")
    print("=" * 65)
    all_types    = sorted(all_type_counts.keys())
    valid_alloys = [a for a in alloy_order if a in stats]
    header_row   = f"{'Type':<30}" + "".join(f"{a:<10}" for a in valid_alloys)
    print(header_row)
    print("-" * 65)

    for ttype in all_types:
        row = f"{ttype:<30}"
        for alloy in valid_alloys:
            count = stats[alloy]["table_types"].get(ttype, 0)
            row  += f"{count:<10}"
        print(row)
    print("=" * 65)

    # ── PDF coverage ───────────────────────────────────────────────────────────
    print("\nPDF COVERAGE")
    print("=" * 40)
    total_papers = totals["total"]
    full_pct     = totals["full_text"]     / max(total_papers, 1) * 100
    abs_pct      = totals["abstract_only"] / max(total_papers, 1) * 100
    print(f"  Papers with full PDF:   {totals['full_text']:>4}  ({full_pct:.1f}%)")
    print(f"  Abstract only:          {totals['abstract_only']:>4}  ({abs_pct:.1f}%)")
    print(f"  PDFs in pdf_map.json:   {len(pdf_map):>4}")
    print("=" * 40)

    # ── Chunk statistics ───────────────────────────────────────────────────────
    print("\nCHUNK STATISTICS")
    print("=" * 40)
    avg_all      = totals["total_chunks"] / max(total_papers, 1)
    avg_full     = totals["total_chunks"] / max(totals["full_text"], 1)
    print(f"  Total chunks:           {totals['total_chunks']:>6}")
    print(f"  Avg chunks per paper:   {avg_all:>6.1f}")
    print(f"  Avg chunks (full text): {avg_full:>6.1f}")
    print("=" * 40)

    # ── ChromaDB verification ──────────────────────────────────────────────────
    print("\nCHROMADB VERIFICATION")
    print("=" * 40)
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path="data/chroma_db",
            settings=Settings(anonymized_telemetry=False),
        )
        col      = client.get_collection("alloyrag")
        expected = totals["total_chunks"] + totals["total_tables"]
        actual   = col.count()
        match    = actual == expected
        print(f"  ChromaDB entries:       {actual:>6}")
        print(f"  Expected (chunks+tables): {expected:>4}")
        print(
            f"  Match: "
            f"{'✅ Yes — index is in sync' if match else '⚠️  No — rebuild index'}"
        )
    except Exception as e:
        print(f"  ChromaDB check failed: {e}")
    print("=" * 40)

    # ── Paper sentence for your methods section ────────────────────────────────
    print("\nREADY-TO-USE TEXT FOR PAPER (Section 3.2)")
    print("=" * 78)
    print(
        f'  "The final corpus comprises {totals["total"]} open-access '
        f'peer-reviewed papers\n'
        f'   ({totals["full_text"]} with full-text PDF extraction, '
        f'{totals["abstract_only"]} abstract-only),\n'
        f'   yielding {totals["total_chunks"]} text chunks and '
        f'{totals["total_tables"]} structured table retrieval units\n'
        f'   spanning the {yr_range} publication period."'
    )
    print("=" * 78)


if __name__ == "__main__":
    get_corpus_stats()