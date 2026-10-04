"""
plot_results.py — READ-ONLY figure generator for AlloyRAG.

===========================  READ-ONLY GUARANTEE  ==========================
This script NEVER modifies source data.

Files opened strictly for READING:
    data/evaluation/evaluation_results.json
    data/evaluation/ablation_results.json
    data/evaluation/ablation_checkpoint.json

Files WRITTEN (new outputs only):
    data/figures/fig1_main_results.{png,pdf}
    data/figures/fig2_k_ablation.{png,pdf}
    data/figures/fig3_retrieval_mode.{png,pdf}
    data/figures/fig4_chunk_content.{png,pdf}
    data/figures/fig5_retrieval_metrics.{png,pdf}
    data/figures/fig6_per_query_faithfulness.{png,pdf}

No other files are created, moved, deleted, or altered.
No network calls, no subprocess, no eval/exec, no pickle.load.

========================  EXACT FIELD SOURCES  ============================
Figure 1 — fig1_main_results
    evaluation_results.json → "ragas_aggregate"
        fields: faithfulness, answer_relevancy,
                context_precision, context_recall
    evaluation_results.json → "bare_llm_ragas"
        fields: faithfulness, answer_relevancy,
                context_precision, context_recall

Figure 2 — fig2_k_ablation
    ablation_results.json → "k_ablation"
        keys:   "k=1", "k=3", "k=5", "k=10"
        fields: faithfulness, answer_relevancy

Figure 3 — fig3_retrieval_mode
    ablation_results.json → "retrieval_mode_ablation"
        keys:   "dense_only", "bm25_only", "hybrid"
        fields: faithfulness, answer_relevancy

Figure 4 — fig4_chunk_content  (two paired sources, 4 bars total)
    [A] ablation_results.json → "chunk_content_ablation"
            keys:   "text_only", "text_and_table"
            fields: faithfulness, answer_relevancy
            (computed over the 7-query stratified subset)
    [B] ablation_checkpoint.json → "chunk_content_ablation"
            keys:   "text_only", "text_and_table"
            per-query entries with "ragas_scored" == True
            fields: faithfulness, answer_relevancy
            (mean computed over every scored query in the checkpoint)

Figure 5 — fig5_retrieval_metrics
    evaluation_results.json → "retrieval_metrics"
        fields: "precision@k" and "recall@k" for every k present

Figure 6 — fig6_per_query_faithfulness
    evaluation_results.json → "per_query_results"
        per-item fields: query_id, difficulty, faithfulness_score

No confidence-interval ranges are drawn anywhere — only exact numbers.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR      = Path("data/figures")
EVAL_PATH       = Path("data/evaluation/evaluation_results.json")
ABLATION_PATH   = Path("data/evaluation/ablation_results.json")
CHECKPOINT_PATH = Path("data/evaluation/ablation_checkpoint.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── IO helpers (read-only) ────────────────────────────────────────────────────

def _read_json(path: Path):
    """Open and parse a JSON file for READING only. Returns None if missing."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    """Write PNG + PDF into data/figures/. The ONLY write action of this script."""
    png = OUTPUT_DIR / f"{name}.png"
    pdf = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"  saved {png.name}  (+ .pdf)")


def _bar_label(ax, bar, value, dy=3, fmt="{:.3f}", fontsize=11):
    if value is None:
        return
    ax.annotate(
        fmt.format(value),
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, dy), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold",
    )


def _point_label(ax, x, y, fmt="{:.3f}", dy=8, color="black", fontsize=11):
    if y is None:
        return
    ax.annotate(
        fmt.format(y), xy=(x, y),
        xytext=(0, dy), textcoords="offset points",
        ha="center",
        fontsize=fontsize, fontweight="bold", color=color,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main results
#   Source: evaluation_results.json
#     ├─ "ragas_aggregate"    → AlloyRAG scores
#     └─ "bare_llm_ragas"     → Bare-LLM baseline scores
# ══════════════════════════════════════════════════════════════════════════════

def fig1_main_results(ev: dict) -> None:
    rag  = ev.get("ragas_aggregate") or {}
    bare = ev.get("bare_llm_ragas")  or {}

    metrics = [
        ("faithfulness",      "Faithfulness"),
        ("answer_relevancy",  "Answer Relevancy"),
        ("context_precision", "Context Precision"),
        ("context_recall",    "Context Recall"),
    ]
    # keep only metrics for which AlloyRAG has a value
    metrics = [(k, l) for k, l in metrics if rag.get(k) is not None]
    if not metrics:
        print("  skip fig1: no 'ragas_aggregate' in evaluation_results.json")
        return

    labels    = [l for _, l in metrics]
    rag_vals  = [rag.get(k)  for k, _ in metrics]
    bare_vals = [bare.get(k) for k, _ in metrics]

    x = np.arange(len(metrics))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    bare_plot = [v if v is not None else 0 for v in bare_vals]
    rag_plot  = [v if v is not None else 0 for v in rag_vals]

    b1 = ax.bar(x - w / 2, bare_plot, w, label="Bare LLM",
                color="#d62728", alpha=0.9)
    b2 = ax.bar(x + w / 2, rag_plot,  w, label="AlloyRAG",
                color="#2ca02c", alpha=0.9)

    for b in b2:
        b.set_edgecolor("black")
        b.set_linewidth(1.2)

    for bar, v in zip(b1, bare_vals):
        _bar_label(ax, bar, v)
    for bar, v in zip(b2, rag_vals):
        _bar_label(ax, bar, v, fontsize=10)

    ax.set_title("AlloyRAG vs Bare LLM — RAGAS metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig1_main_results")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — k ablation
#   Source: ablation_results.json → "k_ablation" → "k={1,3,5,10}"
#   Fields: faithfulness, answer_relevancy
# ══════════════════════════════════════════════════════════════════════════════

def _k_int(k_str: str) -> int:
    try:
        return int(k_str.split("=")[1])
    except (IndexError, ValueError):
        return 10**9


def fig2_k_ablation(abl: dict) -> None:
    k_data = abl.get("k_ablation")
    if not k_data:
        print("  skip fig2: no 'k_ablation' in ablation_results.json")
        return

    keys = sorted(k_data.keys(), key=_k_int)
    k_vals = [_k_int(k) for k in keys]

    faith = [k_data[k].get("faithfulness")     for k in keys]
    relev = [k_data[k].get("answer_relevancy") for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_vals, faith, "o-", color="#1f77b4", lw=2, ms=8,
            label="Faithfulness")
    ax.plot(k_vals, relev, "s-", color="#ff7f0e", lw=2, ms=8,
            label="Answer Relevancy")

    for kv, fv, rv in zip(k_vals, faith, relev):
        _point_label(ax, kv, fv, dy=10, color="#1f77b4")
        _point_label(ax, kv, rv, dy=-16, color="#ff7f0e")

    ax.set_xlabel("Number of retrieved chunks (k)")
    ax.set_ylabel("Score")
    ax.set_title("Effect of retrieved context size on generation quality")
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.set_ylim(0.3, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig2_k_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Retrieval mode ablation
#   Source: ablation_results.json → "retrieval_mode_ablation"
#   Keys:   "dense_only", "bm25_only", "hybrid"
#   Fields: faithfulness, answer_relevancy
# ══════════════════════════════════════════════════════════════════════════════

def fig3_retrieval_mode(abl: dict) -> None:
    data = abl.get("retrieval_mode_ablation")
    if not data:
        print("  skip fig3: no 'retrieval_mode_ablation' in ablation_results.json")
        return

    order  = ["bm25_only", "dense_only", "hybrid"]
    pretty = {"bm25_only": "BM25 only",
              "dense_only": "Dense only",
              "hybrid":     "Hybrid RRF"}

    keys = [k for k in order if k in data]
    if not keys:
        print("  skip fig3: no recognised retrieval conditions")
        return

    labels = [pretty[k] for k in keys]
    faith  = [data[k].get("faithfulness")     for k in keys]
    relev  = [data[k].get("answer_relevancy") for k in keys]

    x = np.arange(len(keys))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    fp = [v if v is not None else 0 for v in faith]
    rp = [v if v is not None else 0 for v in relev]

    b1 = ax.bar(x - w / 2, fp, w, label="Faithfulness",
                color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x + w / 2, rp, w, label="Answer Relevancy",
                color="#ff7f0e", alpha=0.9)

    if "hybrid" in keys:
        hi = keys.index("hybrid")
        for b in (b1[hi], b2[hi]):
            b.set_edgecolor("black")
            b.set_linewidth(1.5)

    for bar, v in zip(b1, faith):
        _bar_label(ax, bar, v, dy=4)
    for bar, v in zip(b2, relev):
        _bar_label(ax, bar, v, dy=4)

    ax.set_ylabel("Score")
    ax.set_title("Retrieval Mode Ablation — BM25 vs Dense vs Hybrid RRF")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.9, loc="upper left")

    fig.tight_layout()
    _save(fig, "fig3_retrieval_mode")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4a — Chunk-content ablation, 7-query stratified subset
#   Source: ablation_results.json → "chunk_content_ablation"
#           (or "chunk_content_ablation_tablerelevant" if present)
#   Keys:   "text_only", "text_and_table"
#   Fields: faithfulness, answer_relevancy
# ══════════════════════════════════════════════════════════════════════════════

def fig4a_chunk_content_subset(abl: dict) -> None:
    res_key = "chunk_content_ablation_tablerelevant"
    if res_key not in abl and "chunk_content_ablation" in abl:
        res_key = "chunk_content_ablation"
    data = abl.get(res_key) or {}
    if not data:
        print("  skip fig4a: no 'chunk_content_ablation' in ablation_results.json")
        return

    order  = ["text_only", "text_and_table"]
    pretty = {"text_only":      "Text only",
              "text_and_table": "Text + Table"}
    keys   = [k for k in order if k in data]
    if not keys:
        print("  skip fig4a: no recognised chunk-content conditions")
        return

    labels = [pretty[k] for k in keys]
    faith  = [data[k].get("faithfulness")     for k in keys]
    relev  = [data[k].get("answer_relevancy") for k in keys]

    n_q = max((data[k].get("n_queries") or 0) for k in keys)

    x = np.arange(len(keys))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    fp = [v if v is not None else 0 for v in faith]
    rp = [v if v is not None else 0 for v in relev]

    b1 = ax.bar(x - w / 2, fp, w, label="Faithfulness",
                color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x + w / 2, rp, w, label="Answer Relevancy",
                color="#ff7f0e", alpha=0.9)

    if "text_and_table" in keys:
        hi = keys.index("text_and_table")
        for b in (b1[hi], b2[hi]):
            b.set_edgecolor("black"); b.set_linewidth(1.5)

    for bar, v in zip(b1, faith):
        _bar_label(ax, bar, v, dy=4)
    for bar, v in zip(b2, relev):
        _bar_label(ax, bar, v, dy=4)

    ax.set_ylabel("Score")
    ax.set_title(f"Chunk Content Ablation — {n_q}-query stratified subset")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig4a_chunk_content_subset")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4b — Chunk-content ablation, full checkpoint set
#   Source: ablation_checkpoint.json → "chunk_content_ablation"
#           per-query entries; mean over every entry with "ragas_scored" == True
#   Fields: faithfulness, answer_relevancy
# ══════════════════════════════════════════════════════════════════════════════

def _mean_from_checkpoint(node: dict):
    """
    node = checkpoint["chunk_content_ablation"]["<condition>"]
    i.e. dict: qid → {ragas_scored, faithfulness, answer_relevancy, ...}
    Returns (mean_faith, mean_relev, n_scored) or (None, None, 0).
    """
    if not isinstance(node, dict):
        return None, None, 0

    faiths, relevs = [], []
    for _qid, entry in node.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("ragas_scored"):
            continue
        fv = entry.get("faithfulness")
        rv = entry.get("answer_relevancy")
        if isinstance(fv, (int, float)) and fv == fv:
            faiths.append(float(fv))
        if isinstance(rv, (int, float)) and rv == rv:
            relevs.append(float(rv))

    mean_f = round(sum(faiths) / len(faiths), 4) if faiths else None
    mean_r = round(sum(relevs) / len(relevs), 4) if relevs else None
    n = max(len(faiths), len(relevs))
    return mean_f, mean_r, n


def fig4b_chunk_content_checkpoint(ckpt: dict) -> None:
    data = ((ckpt or {}).get("chunk_content_ablation")) or {}
    if not data:
        print("  skip fig4b: no 'chunk_content_ablation' in ablation_checkpoint.json")
        return

    order  = ["text_only", "text_and_table"]
    pretty = {"text_only":      "Text only",
              "text_and_table": "Text + Table"}

    labels, faith, relev, n_used = [], [], [], 0
    for key in order:
        node = data.get(key)
        if not node:
            continue
        mf, mr, n = _mean_from_checkpoint(node)
        if mf is None and mr is None:
            continue
        labels.append(pretty[key])
        faith.append(mf)
        relev.append(mr)
        n_used = max(n_used, n)

    if not labels:
        print("  skip fig4b: no scored entries in checkpoint")
        return

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    fp = [v if v is not None else 0 for v in faith]
    rp = [v if v is not None else 0 for v in relev]

    b1 = ax.bar(x - w / 2, fp, w, label="Faithfulness",
                color="#1f77b4", alpha=0.9)
    b2 = ax.bar(x + w / 2, rp, w, label="Answer Relevancy",
                color="#ff7f0e", alpha=0.9)

    if "Text + Table" in labels:
        hi = labels.index("Text + Table")
        for b in (b1[hi], b2[hi]):
            b.set_edgecolor("black"); b.set_linewidth(1.5)

    for bar, v in zip(b1, faith):
        _bar_label(ax, bar, v, dy=4)
    for bar, v in zip(b2, relev):
        _bar_label(ax, bar, v, dy=4)

    ax.set_ylabel("Score")
    ax.set_title(f"Chunk Content Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig4b_chunk_content_checkpoint")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Retrieval precision / recall @ k
#   Source: evaluation_results.json → "retrieval_metrics"
#   Fields: "precision@k" and "recall@k" for every k present
# ══════════════════════════════════════════════════════════════════════════════

def fig5_retrieval_metrics(ev: dict) -> None:
    rm = ev.get("retrieval_metrics") or {}
    if not rm:
        print("  skip fig5: no 'retrieval_metrics' in evaluation_results.json")
        return

    ks = sorted({
        int(k.split("@")[1])
        for k in rm
        if k.startswith("precision@") or k.startswith("recall@")
    })
    if not ks:
        print("  skip fig5: no precision@/recall@ entries")
        return

    precision = [rm.get(f"precision@{k}") for k in ks]
    recall    = [rm.get(f"recall@{k}")    for k in ks]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks, precision, "o-",  color="#1f77b4", lw=2, ms=8,
            label="Precision@k")
    ax.plot(ks, recall,    "s--", color="#d62728", lw=2, ms=8,
            label="Recall@k")

    for k, p, r in zip(ks, precision, recall):
        _point_label(ax, k, p, dy=7,  color="#1f77b4")
        _point_label(ax, k, r, dy=7, color="#d62728")

    ax.set_xlabel("k (number of retrieved chunks)")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval precision and recall at different k")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig5_retrieval_metrics")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Per-query faithfulness (grouped by difficulty)
#   Source: evaluation_results.json → "per_query_results"
#   Per-item fields: query_id, difficulty, faithfulness_score
# ══════════════════════════════════════════════════════════════════════════════

def fig6_per_query_faithfulness(ev: dict) -> None:
    per_q = ev.get("per_query_results") or []
    if not per_q:
        print("  skip fig6: no 'per_query_results' in evaluation_results.json")
        return

    rows = []
    for q in per_q:
        s = q.get("faithfulness_score")
        if s is None:
            continue
        diff = (q.get("difficulty") or "").lower()
        rows.append((q.get("query_id", "?"), diff, float(s)))

    if not rows:
        print("  skip fig6: no faithfulness_score values")
        return

    order = {"easy": 0, "medium": 1, "hard": 2, "": 3, "unknown": 3}
    rows.sort(key=lambda r: (order.get(r[1], 3), r[2]))

    qids   = [r[0] for r in rows]
    diffs  = [r[1] for r in rows]
    scores = [r[2] for r in rows]
    colors = {"easy":    "#2ca02c",
              "medium":  "#ff7f0e",
              "hard":    "#d62728",
              "":        "#7f7f7f",
              "unknown": "#7f7f7f"}

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(rows) + 3), 4.5))
    x = np.arange(len(rows))
    bars = ax.bar(x, scores,
                  color=[colors.get(d, "#7f7f7f") for d in diffs],
                  alpha=0.85, width=0.75)

    for bar, s in zip(bars, scores):
        _bar_label(ax, bar, s, dy=3, fmt="{:.2f}", fontsize=12)

    ax.axhline(0.8, color="gray", ls="--", lw=1, alpha=0.7,
               label="0.80 threshold")
    ax.set_ylabel("Faithfulness score")
    ax.set_title("Per-Query Faithfulness — AlloyRAG")
    ax.set_xticks(x)
    ax.set_xticklabels(qids, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)

    patches = [
        mpatches.Patch(color=colors["easy"],   label="Easy"),
        mpatches.Patch(color=colors["medium"], label="Medium"),
        mpatches.Patch(color=colors["hard"],   label="Hard"),
    ]
    ax.legend(
        handles=patches + [
            plt.Line2D([0], [0], color="gray", ls="--",
                       label="0.80 threshold")
        ],
        framealpha=0.9, loc="upper left", fontsize=9,
    )

    fig.tight_layout()
    _save(fig, "fig6_per_query_faithfulness")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

ALL_FIGURES = {
    "main":        "fig1_main_results",
    "k":           "fig2_k_ablation",
    "retrieval":   "fig3_retrieval_mode",
    "chunk":       "fig4_chunk_content",
    "ret_metrics": "fig5_retrieval_metrics",
    "per_query":   "fig6_per_query_faithfulness",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read-only figure generator for AlloyRAG."
    )
    ap.add_argument(
        "--figures", nargs="+",
        choices=list(ALL_FIGURES.keys()) + ["all"],
        default=["all"],
        help="Which figures to generate.",
    )
    args = ap.parse_args()

    sel = args.figures
    if "all" in sel:
        sel = list(ALL_FIGURES.keys())

    print("Reading source JSONs (read-only)...")
    ev   = _read_json(EVAL_PATH)
    abl  = _read_json(ABLATION_PATH)
    ckpt = _read_json(CHECKPOINT_PATH)

    print(f"  {EVAL_PATH.as_posix()}       : "
          f"{'OK' if ev   is not None else 'MISSING'}")
    print(f"  {ABLATION_PATH.as_posix()}   : "
          f"{'OK' if abl  is not None else 'MISSING'}")
    print(f"  {CHECKPOINT_PATH.as_posix()} : "
          f"{'OK' if ckpt is not None else 'MISSING'}")

    if ev is not None and "per_query_results" in ev:
        print(f"  evaluation_results.json: "
              f"{len(ev.get('per_query_results', []))} per-query rows")

    print(f"\nGenerating {len(sel)} figure(s)...\n")

    if "main"        in sel and ev   is not None: fig1_main_results(ev)
    if "k"           in sel and abl  is not None: fig2_k_ablation(abl)
    if "retrieval"   in sel and abl  is not None: fig3_retrieval_mode(abl)
    if "chunk"       in sel and abl  is not None: fig4a_chunk_content_subset(abl)
    if "chunk"       in sel and ckpt is not None: fig4b_chunk_content_checkpoint(ckpt)
    if "ret_metrics" in sel and ev   is not None: fig5_retrieval_metrics(ev)
    if "per_query"   in sel and ev   is not None: fig6_per_query_faithfulness(ev)

    print(f"\n✅ Figures written to: {OUTPUT_DIR.resolve()}")
    print("   Source data files were only READ; nothing was modified.")


if __name__ == "__main__":
    main()