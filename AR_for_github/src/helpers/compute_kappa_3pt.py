# src/helpers/compute_kappa_3pt.py
"""
Cohen's kappa computation for the 3-point expert rubric:
    Poor / Adequate / Good

Reads from two expert CSVs. Accepts either:
  - numeric scores 1 / 2 / 3
  - word scores "Poor" / "Adequate" / "Good" (case-insensitive)

Read-only. Overwrites nothing. Prints a report.
"""

import csv
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

EXPERT1_CSV = Path("data/evaluation/expert_annotation_template_3pt.csv")
EXPERT2_CSV = Path("data/evaluation/expert_annotation_expert2_3pt.csv")

# Column names expected in the CSVs
COL_RAG  = "expert_score_rag_3pt"
COL_BARE = "expert_score_bare_3pt"

# Map words → numeric (1=Poor, 2=Adequate, 3=Good)
WORD_MAP = {
    "poor":     1,
    "adequate": 2,
    "good":     3,
    # tolerated synonyms
    "bad":      1,
    "fair":     2,
    "ok":       2,
    "okay":     2,
    "great":    3,
    "excellent": 3,
}


def parse_score(raw: str) -> int | None:
    """Parse a single cell into 1/2/3, or None if invalid/empty."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Numeric path
    try:
        n = int(float(s))
        if 1 <= n <= 3:
            return n
        print(f"   ⚠️ Out-of-range numeric '{s}' (must be 1–3) — skipped")
        return None
    except ValueError:
        pass

    # Word path
    key = s.lower()
    if key in WORD_MAP:
        return WORD_MAP[key]

    print(f"   ⚠️ Unrecognised score '{s}' (expected Poor/Adequate/Good or 1–3) — skipped")
    return None


def load_scores(path: Path, col: str) -> dict[str, int]:
    """Return {query_id: 1/2/3} for rows with a valid value."""
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"❌ File is empty: {path}")
        sys.exit(1)

    if col not in rows[0]:
        print(f"❌ Column '{col}' not found in {path}")
        print(f"   Available columns: {list(rows[0].keys())}")
        sys.exit(1)

    out: dict[str, int] = {}
    for r in rows:
        qid = r.get("query_id", "").strip()
        if not qid:
            continue
        val = parse_score(r.get(col, ""))
        if val is not None:
            out[qid] = val
    return out


def align(e1: dict, e2: dict, label: str):
    """Return aligned (list1, list2, shared_qids) restricted to shared query_ids."""
    shared = sorted(set(e1) & set(e2))
    if not shared:
        print(f"❌ No shared query_ids between experts for '{label}'")
        sys.exit(1)

    only_e1 = set(e1) - set(e2)
    only_e2 = set(e2) - set(e1)
    if only_e1:
        print(f"   ⚠️ {label}: only expert 1 rated {sorted(only_e1)}")
    if only_e2:
        print(f"   ⚠️ {label}: only expert 2 rated {sorted(only_e2)}")

    return [e1[q] for q in shared], [e2[q] for q in shared], shared


def interpret(k: float) -> str:
    """Landis & Koch (1977) interpretation — scale-agnostic."""
    if k >= 0.80:  return "almost perfect agreement ✅"
    if k >= 0.60:  return "substantial agreement ✅"
    if k >= 0.40:  return "moderate agreement — acceptable"
    if k >= 0.20:  return "fair agreement — discuss differences"
    return "poor agreement — review rubric with experts"


def label_counts(scores: list[int]) -> dict[int, int]:
    """Count how many 1/2/3 in a list."""
    return {1: scores.count(1), 2: scores.count(2), 3: scores.count(3)}


def main():
    for path in (EXPERT1_CSV, EXPERT2_CSV):
        if not path.exists():
            print(f"❌ Missing file: {path}")
            print(f"   Expected at: {path.resolve()}")
            print(f"   Run this script from the project root (AlloyRAG/).")
            sys.exit(1)

    # Load both experts for both systems
    e1_rag  = load_scores(EXPERT1_CSV, COL_RAG)
    e2_rag  = load_scores(EXPERT2_CSV, COL_RAG)
    e1_bare = load_scores(EXPERT1_CSV, COL_BARE)
    e2_bare = load_scores(EXPERT2_CSV, COL_BARE)

    r1, r2, r_qids = align(e1_rag,  e2_rag,  "RAG")
    b1, b2, b_qids = align(e1_bare, e2_bare, "Bare")

    print()
    print("=" * 60)
    print("  COHEN'S KAPPA  —  3-POINT SCALE (Poor/Adequate/Good)")
    print("=" * 60)
    print(f"  Aligned queries — RAG:   {len(r_qids)}")
    print(f"  Aligned queries — Bare:  {len(b_qids)}")
    print()

    # Unweighted and linear-weighted
    kappa_rag_unw  = cohen_kappa_score(r1, r2)
    kappa_rag_lin  = cohen_kappa_score(r1, r2, weights="linear")
    kappa_bare_unw = cohen_kappa_score(b1, b2)
    kappa_bare_lin = cohen_kappa_score(b1, b2, weights="linear")

    print("  Cohen's kappa (RAG):")
    print(f"      unweighted : {kappa_rag_unw:.3f}")
    print(f"      linear     : {kappa_rag_lin:.3f}")
    print("  Cohen's kappa (Bare LLM):")
    print(f"      unweighted : {kappa_bare_unw:.3f}")
    print(f"      linear     : {kappa_bare_lin:.3f}")
    print()

    # Means
    mean_rag_e1  = sum(r1) / len(r1)
    mean_rag_e2  = sum(r2) / len(r2)
    mean_bare_e1 = sum(b1) / len(b1)
    mean_bare_e2 = sum(b2) / len(b2)

    print(f"  Mean RAG score  (Expert 1): {mean_rag_e1:.2f}/3   (n={len(r1)})")
    print(f"  Mean RAG score  (Expert 2): {mean_rag_e2:.2f}/3   (n={len(r2)})")
    print(f"  Mean Bare score (Expert 1): {mean_bare_e1:.2f}/3   (n={len(b1)})")
    print(f"  Mean Bare score (Expert 2): {mean_bare_e2:.2f}/3   (n={len(b2)})")
    print()
    print(f"  Combined mean RAG:  {(mean_rag_e1 + mean_rag_e2) / 2:.2f}/3")
    print(f"  Combined mean Bare: {(mean_bare_e1 + mean_bare_e2) / 2:.2f}/3")
    print()

    # Distribution of ratings
    print("  Rating distribution:")
    print(f"      RAG  (E1): Poor={label_counts(r1)[1]}  Adequate={label_counts(r1)[2]}  Good={label_counts(r1)[3]}")
    print(f"      RAG  (E2): Poor={label_counts(r2)[1]}  Adequate={label_counts(r2)[2]}  Good={label_counts(r2)[3]}")
    print(f"      Bare (E1): Poor={label_counts(b1)[1]}  Adequate={label_counts(b1)[2]}  Good={label_counts(b1)[3]}")
    print(f"      Bare (E2): Poor={label_counts(b2)[1]}  Adequate={label_counts(b2)[2]}  Good={label_counts(b2)[3]}")
    print()

    print("─" * 60)
    print(f"  Kappa RAG  (linear): {kappa_rag_lin:.3f} — {interpret(kappa_rag_lin)}")
    print(f"  Kappa Bare (linear): {kappa_bare_lin:.3f} — {interpret(kappa_bare_lin)}")
    print("=" * 60)


if __name__ == "__main__":
    main()