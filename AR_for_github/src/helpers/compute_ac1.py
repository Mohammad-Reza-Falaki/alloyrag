# src/helpers/compute_ac1.py
"""
Gwet's AC1 for the 3-point expert rubric (Poor / Adequate / Good).

AC1 was proposed by Gwet (2008) specifically to address the kappa paradox:
when raters have skewed marginals (e.g., one rater almost always picks "Good"),
Cohen's kappa collapses toward zero or negative even when observed agreement is high.

Formula (2 raters, q categories):
    P_a   = observed agreement (proportion of items where raters agree)
    p_k   = proportion of ALL ratings that fall in category k
    P_e   = (1 / (q-1)) * Σ_k p_k * (1 - p_k)      ← Gwet's chance agreement
    AC1   = (P_a - P_e) / (1 - P_e)

Read-only. Prints a report.
"""

import csv
import sys
from pathlib import Path

EXPERT1_CSV = Path("data/evaluation/expert_annotation_template_3pt.csv")
EXPERT2_CSV = Path("data/evaluation/expert_annotation_expert2_3pt.csv")

COL_RAG  = "expert_score_rag_3pt"
COL_BARE = "expert_score_bare_3pt"

WORD_MAP = {
    "poor": 1, "bad": 1,
    "adequate": 2, "fair": 2, "ok": 2, "okay": 2,
    "good": 3, "great": 3, "excellent": 3,
}


def parse_score(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        n = int(float(s))
        return n if 1 <= n <= 3 else None
    except ValueError:
        return WORD_MAP.get(s.lower())


def load_scores(path: Path, col: str) -> dict:
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or col not in rows[0]:
        print(f"❌ Missing column '{col}' in {path}")
        sys.exit(1)
    out = {}
    for r in rows:
        qid = r.get("query_id", "").strip()
        if not qid:
            continue
        v = parse_score(r.get(col, ""))
        if v is not None:
            out[qid] = v
    return out


def align(e1: dict, e2: dict, label: str):
    shared = sorted(set(e1) & set(e2))
    if not shared:
        print(f"❌ No shared query_ids for {label}")
        sys.exit(1)
    return [e1[q] for q in shared], [e2[q] for q in shared], shared


# ══════════════════════════════════════════════════════════════════════════════
# GWET'S AC1
# ══════════════════════════════════════════════════════════════════════════════

def gwet_ac1(r1: list, r2: list, q: int = 3) -> float:
    """
    Gwet's AC1 for 2 raters and q nominal categories.
    r1, r2 are equal-length lists of integer categories (1..q).
    """
    n = len(r1)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(1 for a, b in zip(r1, r2) if a == b)
    p_a = agree / n

    # Marginal proportions across ALL ratings (both raters)
    total = 2 * n
    counts = {k: 0 for k in range(1, q + 1)}
    for v in r1 + r2:
        counts[v] += 1
    p_k = {k: counts[k] / total for k in range(1, q + 1)}

    # Gwet chance agreement
    p_e = (1.0 / (q - 1)) * sum(p_k[k] * (1 - p_k[k]) for k in range(1, q + 1))

    if p_e >= 1.0:
        return 1.0
    return (p_a - p_e) / (1.0 - p_e)


# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE: COHEN'S KAPPA + KRIPPENDORFF-STYLE OBSERVED AGREEMENT
# ══════════════════════════════════════════════════════════════════════════════

def cohen_kappa(r1: list, r2: list, q: int = 3) -> float:
    n = len(r1)
    if n == 0:
        return 0.0
    p_a = sum(1 for a, b in zip(r1, r2) if a == b) / n
    counts_1 = {k: 0 for k in range(1, q + 1)}
    counts_2 = {k: 0 for k in range(1, q + 1)}
    for v in r1:
        counts_1[v] += 1
    for v in r2:
        counts_2[v] += 1
    p_e = sum((counts_1[k] / n) * (counts_2[k] / n) for k in range(1, q + 1))
    if p_e >= 1.0:
        return 1.0
    return (p_a - p_e) / (1.0 - p_e)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def interpret(k: float) -> str:
    if k >= 0.80: return "almost perfect agreement ✅"
    if k >= 0.60: return "substantial agreement ✅"
    if k >= 0.40: return "moderate agreement — acceptable"
    if k >= 0.20: return "fair agreement — discuss differences"
    return "poor agreement — review rubric with experts"


def rating_distribution(scores: list) -> dict:
    return {1: scores.count(1), 2: scores.count(2), 3: scores.count(3)}


def main():
    for p in (EXPERT1_CSV, EXPERT2_CSV):
        if not p.exists():
            print(f"❌ Missing file: {p}")
            sys.exit(1)

    e1_rag  = load_scores(EXPERT1_CSV, COL_RAG)
    e2_rag  = load_scores(EXPERT2_CSV, COL_RAG)
    e1_bare = load_scores(EXPERT1_CSV, COL_BARE)
    e2_bare = load_scores(EXPERT2_CSV, COL_BARE)

    r1, r2, r_qids = align(e1_rag,  e2_rag,  "RAG")
    b1, b2, b_qids = align(e1_bare, e2_bare, "Bare")

    print()
    print("=" * 68)
    print("  AGREEMENT METRICS — 3-POINT SCALE (Poor / Adequate / Good)")
    print("=" * 68)
    print(f"  Aligned queries — RAG:   {len(r_qids)}")
    print(f"  Aligned queries — Bare:  {len(b_qids)}")
    print()

    # ── RAG ───────────────────────────────────────────────────────────────
    ac1_rag   = gwet_ac1(r1, r2)
    kap_rag   = cohen_kappa(r1, r2)
    p_a_rag   = sum(1 for a, b in zip(r1, r2) if a == b) / len(r1)

    print("  ─── RAG ───")
    print(f"      Observed agreement P_a : {p_a_rag:.3f}")
    print(f"      Cohen's kappa          : {kap_rag:.3f}  ({interpret(kap_rag)})")
    print(f"      Gwet's AC1             : {ac1_rag:.3f}  ({interpret(ac1_rag)})")
    print()

    # ── Bare ──────────────────────────────────────────────────────────────
    ac1_bare  = gwet_ac1(b1, b2)
    kap_bare  = cohen_kappa(b1, b2)
    p_a_bare  = sum(1 for a, b in zip(b1, b2) if a == b) / len(b1)

    print("  ─── Bare LLM ───")
    print(f"      Observed agreement P_a : {p_a_bare:.3f}")
    print(f"      Cohen's kappa          : {kap_bare:.3f}  ({interpret(kap_bare)})")
    print(f"      Gwet's AC1             : {ac1_bare:.3f}  ({interpret(ac1_bare)})")
    print()

    # ── Rating distributions (why kappa broke) ────────────────────────────
    print("  ─── Rating distributions ───")
    for name, scores in (("RAG  E1", r1), ("RAG  E2", r2),
                         ("Bare E1", b1), ("Bare E2", b2)):
        d = rating_distribution(scores)
        print(f"      {name}: Poor={d[1]}  Adequate={d[2]}  Good={d[3]}")
    print()

    # ── Interpretation ────────────────────────────────────────────────────
    print("─" * 68)
    print(f"  RAG:  Cohen κ={kap_rag:.3f}  |  Gwet AC1={ac1_rag:.3f}")
    print(f"  Bare: Cohen κ={kap_bare:.3f}  |  Gwet AC1={ac1_bare:.3f}")
    print()

    if ac1_rag - kap_rag > 0.15:
        print("  Note: AC1 >> kappa for RAG — prevalence effect detected.")
        print("        Skewed marginals are inflating chance agreement under kappa.")
    if ac1_bare - kap_bare > 0.15:
        print("  Note: AC1 >> kappa for Bare — severe prevalence effect.")
        print("        Expert 2 rated almost everything Good on the Bare LLM.")
    print("=" * 68)


if __name__ == "__main__":
    main()