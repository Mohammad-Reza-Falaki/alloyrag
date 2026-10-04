# src/helpers/compute_kappa.py
import csv
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

EXPERT1_CSV = Path("data/evaluation/expert_annotation_template.csv")
EXPERT2_CSV = Path("data/evaluation/expert_annotation_expert2.csv")


def load_scores(path: Path, col: str) -> dict[str, int]:
    """Return {query_id: score} for rows with a non-empty value."""
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

    out = {}
    for r in rows:
        qid = r.get("query_id", "").strip()
        val = r.get(col, "").strip()
        if not qid or not val:
            continue
        try:
            out[qid] = int(float(val))
        except ValueError:
            print(f"   ⚠️ Skipping non-numeric '{val}' for {qid} in {path.name}")
    return out


def align(e1: dict, e2: dict, label: str):
    """Return aligned (list1, list2) restricted to shared query_ids."""
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


def main():
    for path in (EXPERT1_CSV, EXPERT2_CSV):
        if not path.exists():
            print(f"❌ Missing file: {path}")
            print(f"   Expected at: {path.resolve()}")
            print(f"   Run this script from the project root (AlloyRAG/).")
            sys.exit(1)

    e1_rag  = load_scores(EXPERT1_CSV, "expert_score_rag_1_5")
    e2_rag  = load_scores(EXPERT2_CSV, "expert_score_rag_1_5")
    e1_bare = load_scores(EXPERT1_CSV, "expert_score_bare_1_5")
    e2_bare = load_scores(EXPERT2_CSV, "expert_score_bare_1_5")

    r1, r2, r_qids = align(e1_rag,  e2_rag,  "RAG")
    b1, b2, b_qids = align(e1_bare, e2_bare, "Bare")

    print()
    print(f"Aligned queries — RAG:  {len(r_qids)}")
    print(f"Aligned queries — Bare: {len(b_qids)}")
    print()

    kappa_rag  = cohen_kappa_score(r1, r2, weights="linear")
    kappa_bare = cohen_kappa_score(b1, b2, weights="linear")

    print(f"Cohen's kappa (RAG):      {kappa_rag:.3f}")
    print(f"Cohen's kappa (Bare LLM): {kappa_bare:.3f}")
    print()

    mean_rag_e1  = sum(r1) / len(r1)
    mean_rag_e2  = sum(r2) / len(r2)
    mean_bare_e1 = sum(b1) / len(b1)
    mean_bare_e2 = sum(b2) / len(b2)

    print(f"Mean RAG score  (Expert 1): {mean_rag_e1:.2f}/5   (n={len(r1)})")
    print(f"Mean RAG score  (Expert 2): {mean_rag_e2:.2f}/5   (n={len(r2)})")
    print(f"Mean Bare score (Expert 1): {mean_bare_e1:.2f}/5   (n={len(b1)})")
    print(f"Mean Bare score (Expert 2): {mean_bare_e2:.2f}/5   (n={len(b2)})")
    print()
    print(f"Combined mean RAG:  {(mean_rag_e1 + mean_rag_e2) / 2:.2f}/5")
    print(f"Combined mean Bare: {(mean_bare_e1 + mean_bare_e2) / 2:.2f}/5")
    print()

    def interpret(k: float, label: str) -> None:
        if k >= 0.8:   tag = "almost perfect agreement ✅"
        elif k >= 0.6: tag = "substantial agreement ✅"
        elif k >= 0.4: tag = "moderate agreement — acceptable"
        elif k >= 0.2: tag = "fair agreement — discuss differences"
        else:          tag = "poor agreement — review rubric with experts"
        print(f"{label}: {k:.3f} — {tag}")

    interpret(kappa_rag,  "Kappa RAG ")
    interpret(kappa_bare, "Kappa Bare")


if __name__ == "__main__":
    main()