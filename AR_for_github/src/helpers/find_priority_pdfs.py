"""
find_priority_pdfs.py
Automatically matches missing PDFs against your 20 gold QA queries
using keyword similarity — tells you exactly which ones to download manually
Save as: src/find_priority_pdfs.py
"""

import json
import re
from pathlib import Path

MANUAL_NEEDED_PATH = Path("data/raw_metadata/manual_download_needed.json")
GOLD_QA_PATH       = Path("data/gold_qa.json")
OUTPUT_PATH        = Path("data/raw_metadata/priority_download_list.json")

# ── Keyword extraction from query text ────────────────────────────────────────

def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords from a query or title.
    Removes stopwords and short words.
    """
    stopwords = {
        "a", "an", "the", "and", "or", "of", "in", "for", "to", "with",
        "on", "at", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "what", "which", "how", "why", "when",
        "where", "who", "i", "we", "you", "that", "this", "it", "its",
        "need", "want", "good", "best", "better", "high", "low", "above",
        "below", "than", "more", "less", "very", "also", "both", "between",
    }

    words = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in stopwords}


def compute_match_score(query_keywords: set, title_keywords: set) -> float:
    """
    Simple Jaccard-like overlap score between query and title keywords.
    Returns 0.0 to 1.0
    """
    if not query_keywords or not title_keywords:
        return 0.0
    intersection = query_keywords & title_keywords
    union = query_keywords | title_keywords
    return len(intersection) / len(union)


def find_priority_downloads() -> None:

    # ── Load data ──────────────────────────────────────────────────────────────
    if not MANUAL_NEEDED_PATH.exists():
        print("❌ manual_download_needed.json not found")
        print("   Run collect_papers.py first")
        return

    if not GOLD_QA_PATH.exists():
        print("❌ gold_qa.json not found")
        return

    with open(MANUAL_NEEDED_PATH) as f:
        missing_papers = json.load(f)

    with open(GOLD_QA_PATH) as f:
        gold_qa = json.load(f)

    print(f"Missing PDFs:     {len(missing_papers)}")
    print(f"Gold QA queries:  {len(gold_qa)}")
    print()

    # ── For each missing paper, find best matching QA query ───────────────────
    results = []

    for paper in missing_papers:
        title = paper.get("title", "")
        if not title:
            continue

        title_keywords = extract_keywords(title)
        best_score     = 0.0
        best_query_id  = ""
        best_query     = ""
        best_alloy     = ""

        for qa in gold_qa:
            query          = qa.get("query", "")
            ground_truth   = qa.get("ground_truth_answer", "")
            combined_text  = query + " " + ground_truth
            query_keywords = extract_keywords(combined_text)

            score = compute_match_score(query_keywords, title_keywords)

            if score > best_score:
                best_score    = score
                best_query_id = qa.get("query_id", "")
                best_query    = query
                best_alloy    = qa.get("alloy_system", "")

        results.append({
            "priority_score":  round(best_score, 3),
            "paper_id":        paper.get("paper_id", ""),
            "title":           title,
            "doi":             paper.get("doi", ""),
            "journal":         paper.get("journal", ""),
            "best_match_query_id":  best_query_id,
            "best_match_query":     best_query[:100],
            "alloy_system":         best_alloy,
        })

    # ── Sort by priority score descending ─────────────────────────────────────
    results.sort(key=lambda x: x["priority_score"], reverse=True)

    # ── Save full ranked list ──────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print top 20 to terminal ───────────────────────────────────────────────
    print("=" * 70)
    print("TOP PRIORITY PDFs TO DOWNLOAD MANUALLY")
    print("(ranked by relevance to your 20 gold QA queries)")
    print("=" * 70)

    high_priority   = [r for r in results if r["priority_score"] >= 0.15]
    medium_priority = [r for r in results if 0.08 <= r["priority_score"] < 0.15]
    low_priority    = [r for r in results if r["priority_score"] < 0.08]

    print(f"\n🔴 HIGH PRIORITY — download these first ({len(high_priority)} papers)")
    print("   These directly match your gold QA queries\n")
    for i, r in enumerate(high_priority[:15]):
        print(f"  {i+1:2}. [{r['alloy_system']}] Score: {r['priority_score']:.3f}")
        print(f"      Title:   {r['title'][:70]}")
        print(f"      DOI:     {r['doi']}")
        print(f"      Matches: {r['best_match_query_id']} — {r['best_match_query'][:60]}")
        print()

    print(f"\n🟡 MEDIUM PRIORITY ({len(medium_priority)} papers)")
    print("   Partially related — download if time allows\n")
    for i, r in enumerate(medium_priority[:10]):
        print(f"  {i+1:2}. [{r['alloy_system']}] Score: {r['priority_score']:.3f}")
        print(f"      Title: {r['title'][:70]}")
        print(f"      DOI:   {r['doi']}")
        print()

    print(f"\n⚪ LOW PRIORITY ({len(low_priority)} papers)")
    print("   Abstract-only is fine for these — skip manual download\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"RECOMMENDATION:")
    print(f"  Download HIGH priority:   {min(len(high_priority), 15)} papers")
    print(f"  Download MEDIUM priority: {min(len(medium_priority), 5)} papers  (optional)")
    print(f"  Skip LOW priority:        {len(low_priority)} papers")
    print(f"\nFull ranked list saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    find_priority_downloads()