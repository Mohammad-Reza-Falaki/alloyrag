# src/helpers/inspect_chromadb.py
"""
Inspects ChromaDB index contents and verifies alignment with corpus.
Run after build_index.py and rebuild_idmap.py to confirm everything is in sync.
Run: python src/helpers/inspect_chromadb.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
from collections import Counter
import chromadb
from chromadb.config import Settings


CHROMA_DB_PATH = "data/chroma_db"
BM25_PATH      = "data/bm25_index.pkl"
IDMAP_PATH     = "data/id_to_text.pkl"
CORPUS_DIR     = Path("data/corpus")


def inspect_chromadb():

    print("=" * 60)
    print("CHROMADB INDEX INSPECTION")
    print("=" * 60)

    # ── Connect to ChromaDB ────────────────────────────────────────────────────
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection("alloyrag")
    except Exception as e:
        print(f"❌ Could not connect to ChromaDB: {e}")
        print("   Make sure you ran build_index.py first")
        return

    total_entries = col.count()
    print(f"\nTotal ChromaDB entries: {total_entries}")

    # ── Fetch all metadata ─────────────────────────────────────────────────────
    print("\nFetching metadata from all entries...")
    try:
        results = col.get(include=["metadatas"])
        metas   = results["metadatas"]
        ids     = results["ids"]
    except Exception as e:
        print(f"❌ Could not fetch metadata: {e}")
        return

    # ── Breakdown by chunk_type ────────────────────────────────────────────────
    type_counts = Counter(m.get("chunk_type", "unknown") for m in metas)
    print("\nBy chunk type:")
    print("-" * 35)
    for ctype, count in sorted(type_counts.items()):
        pct = count / max(total_entries, 1) * 100
        print(f"  {ctype:<15} {count:>6}  ({pct:.1f}%)")

    # ── Breakdown by alloy system ──────────────────────────────────────────────
    alloy_counts = Counter(m.get("alloy_system", "Unknown") for m in metas)
    print("\nBy alloy system:")
    print("-" * 35)
    for alloy, count in sorted(
        alloy_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = count / max(total_entries, 1) * 100
        print(f"  {alloy:<15} {count:>6}  ({pct:.1f}%)")

    # ── Breakdown by section ───────────────────────────────────────────────────
    section_counts = Counter(m.get("section", "unknown") for m in metas)
    print("\nBy section (top 10):")
    print("-" * 40)
    for section, count in section_counts.most_common(10):
        print(f"  {section:<30} {count:>6}")

    # ── Year distribution ──────────────────────────────────────────────────────
    years = [m.get("year", 0) for m in metas if m.get("year")]
    if years:
        print(f"\nYear range: {min(years)} – {max(years)}")
        year_counts = Counter(years)
        recent      = sorted(year_counts.items())[-5:]
        print("Most recent years:")
        for yr, count in sorted(recent, reverse=True):
            print(f"  {yr}: {count} entries")

    # ── Unique papers ──────────────────────────────────────────────────────────
    unique_papers = set(m.get("paper_id", "") for m in metas)
    print(f"\nUnique papers indexed: {len(unique_papers)}")

    # ── Sample entries ─────────────────────────────────────────────────────────
    print("\nSample TEXT entry:")
    print("-" * 50)
    text_ids = [
        ids[i] for i, m in enumerate(metas)
        if m.get("chunk_type") == "text"
    ]
    if text_ids:
        sample = col.get(
            ids=[text_ids[0]],
            include=["documents", "metadatas"],
        )
        print(f"  ID:       {text_ids[0]}")
        print(f"  Metadata: {sample['metadatas'][0]}")
        print(f"  Text:     {sample['documents'][0][:200]}...")

    print("\nSample TABLE entry:")
    print("-" * 50)
    table_ids = [
        ids[i] for i, m in enumerate(metas)
        if m.get("chunk_type") == "table"
    ]
    if table_ids:
        sample = col.get(
            ids=[table_ids[0]],
            include=["documents", "metadatas"],
        )
        print(f"  ID:       {table_ids[0]}")
        print(f"  Metadata: {sample['metadatas'][0]}")
        print(f"  Text:     {sample['documents'][0][:200]}...")
    else:
        print("  No table entries found")

    # ── BM25 index check ───────────────────────────────────────────────────────
    print("\nBM25 INDEX CHECK")
    print("-" * 35)
    if Path(BM25_PATH).exists():
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        bm25_count = len(bm25_data.get("ids", []))
        print(f"  BM25 entries: {bm25_count}")
        match_bm25 = bm25_count == total_entries
        print(
            f"  Match ChromaDB: "
            f"{'✅ Yes' if match_bm25 else '⚠️  No — rebuild index'}"
        )
    else:
        print("  ❌ bm25_index.pkl not found — run build_index.py")

    # ── ID map check ───────────────────────────────────────────────────────────
    print("\nID MAP CHECK")
    print("-" * 35)
    if Path(IDMAP_PATH).exists():
        with open(IDMAP_PATH, "rb") as f:
            mapping     = pickle.load(f)
        id_to_text  = mapping.get("id_to_text", {})
        id_to_meta  = mapping.get("id_to_meta", {})
        idmap_count = len(id_to_text)
        print(f"  id_to_text entries: {idmap_count}")
        print(f"  id_to_meta entries: {len(id_to_meta)}")
        match_idmap = idmap_count == total_entries
        print(
            f"  Match ChromaDB: "
            f"{'✅ Yes' if match_idmap else '⚠️  No — run rebuild_idmap.py'}"
        )
    else:
        print("  ❌ id_to_text.pkl not found — run rebuild_idmap.py")

    # ── Corpus alignment check ─────────────────────────────────────────────────
    print("\nCORPUS ALIGNMENT CHECK")
    print("-" * 35)
    corpus_files = list(CORPUS_DIR.glob("*.json"))
    corpus_count = len(corpus_files)
    print(f"  Corpus JSON files: {corpus_count}")
    print(f"  Unique papers in ChromaDB: {len(unique_papers)}")
    match_corpus = len(unique_papers) == corpus_count
    print(
        f"  Match: "
        f"{'✅ Yes' if match_corpus else '⚠️  Mismatch — some papers may not be indexed'}"
    )

    if not match_corpus:
        # Find which corpus files are missing from ChromaDB
        corpus_ids = {p.stem for p in corpus_files}
        missing    = corpus_ids - unique_papers
        extra      = unique_papers - corpus_ids
        if missing:
            print(f"\n  Papers in corpus but NOT in ChromaDB ({len(missing)}):")
            for pid in sorted(missing)[:10]:
                print(f"    {pid}")
            if len(missing) > 10:
                print(f"    ... and {len(missing)-10} more")
        if extra:
            print(f"\n  Papers in ChromaDB but NOT in corpus ({len(extra)}):")
            for pid in sorted(extra)[:5]:
                print(f"    {pid}")

    # ── Final verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    all_good = (
        Path(BM25_PATH).exists() and
        Path(IDMAP_PATH).exists() and
        match_corpus and
        (bm25_count == total_entries if Path(BM25_PATH).exists() else False) and
        (idmap_count == total_entries if Path(IDMAP_PATH).exists() else False)
    )

    if all_good:
        print("✅ All indexes are in sync — safe to run rag_pipeline.py")
    else:
        print("⚠️  Issues found — fix before running evaluate.py:")
        if not Path(BM25_PATH).exists():
            print("   → Run: python src/build_index.py")
        elif bm25_count != total_entries:
            print("   → Run: python src/build_index.py")
        if not Path(IDMAP_PATH).exists():
            print("   → Run: python src/helpers/rebuild_idmap.py")
        elif idmap_count != total_entries:
            print("   → Run: python src/helpers/rebuild_idmap.py")
        if not match_corpus:
            print("   → Run: python src/build_index.py (new papers need indexing)")

    print("=" * 60)


if __name__ == "__main__":
    inspect_chromadb()