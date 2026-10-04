#!/usr/bin/env python3
"""
audit_tables.py — READ-ONLY table extraction quality audit.

Scans data/corpus/*.json and reports:
  - Total tables
  - Quality score per table (0-100)
  - Distribution of quality
  - Worst 20 tables with samples
  - Common failure patterns

Usage:
    python src/helpers/audit_tables.py
    python src/helpers/audit_tables.py --dump 5   # print 5 worst tables in full
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter

CORPUS_DIR = Path("data/corpus")

# ── Quality signals ────────────────────────────────────────────────────────────

# Mojibake / non-ASCII-heavy headers
ARABIC_RE   = re.compile(r"[\u0600-\u06FF]")
CJK_RE      = re.compile(r"[\u4E00-\u9FFF]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

# Placeholder column names
PLACEHOLDER_RE = re.compile(r"\bcol_\d+\b", re.IGNORECASE)

# Junk alloy designations (years, page numbers, volume numbers)
YEAR_RE = re.compile(r"^(19|20)\d{2}$")
PAGE_RE = re.compile(r"^\d{3,5}$")


def is_ascii_clean(s: str) -> bool:
    """Return True if the string is mostly ASCII printable."""
    if not s:
        return False
    ascii_chars = sum(1 for c in s if 32 <= ord(c) < 127)
    return ascii_chars / max(len(s), 1) > 0.85


def score_table(tbl: dict) -> tuple[int, list[str]]:
    """
    Return (score 0-100, list of failure reasons).
    Higher is better.
    """
    score = 100
    reasons = []

    header = tbl.get("header", [])
    rows   = tbl.get("rows", [])
    caption = tbl.get("caption", "")
    trepr   = tbl.get("text_representation", "")

    # ── Header quality ────────────────────────────────────────────────────────
    if len(header) < 2:
        score -= 20
        reasons.append("header_too_short")

    non_ascii = sum(1 for h in header if not is_ascii_clean(str(h)))
    if non_ascii > 0:
        score -= min(30, non_ascii * 5)
        reasons.append(f"{non_ascii}_non_ascii_headers")

    long_headers = sum(1 for h in header if len(str(h)) > 60)
    if long_headers > 0:
        score -= min(15, long_headers * 3)
        reasons.append(f"{long_headers}_long_headers")

    # Placeholder column names
    joined_header = " ".join(str(h) for h in header)
    placeholders = len(PLACEHOLDER_RE.findall(joined_header))
    if placeholders > 0:
        score -= min(20, placeholders * 3)
        reasons.append(f"{placeholders}_placeholder_columns")

    # ── Row quality ───────────────────────────────────────────────────────────
    if len(rows) < 2:
        score -= 20
        reasons.append("too_few_rows")

    # Check for empty cells
    empty_cells = 0
    total_cells = 0
    for r in rows:
        for v in r.values():
            total_cells += 1
            if not v or str(v).strip() in ("", "-", "—", "N/A"):
                empty_cells += 1
    if total_cells > 0:
        empty_ratio = empty_cells / total_cells
        if empty_ratio > 0.4:
            score -= int(empty_ratio * 30)
            reasons.append(f"empty_ratio_{empty_ratio:.2f}")

    # ── Caption quality ───────────────────────────────────────────────────────
    if not caption or len(caption.strip()) < 10:
        score -= 10
        reasons.append("no_caption")

    # ── Text representation quality ───────────────────────────────────────────
    if not trepr or len(trepr) < 50:
        score -= 15
        reasons.append("empty_text_representation")
    else:
        # Look for obvious garbage in the text representation
        if ARABIC_RE.search(trepr) or CJK_RE.search(trepr) or CYRILLIC_RE.search(trepr):
            score -= 20
            reasons.append("non_latin_in_representation")
        if not is_ascii_clean(trepr):
            score -= 10
            reasons.append("dirty_representation")

    return max(0, score), reasons


def audit_corpus(dump_n: int = 0) -> None:
    files = sorted(CORPUS_DIR.glob("*.json"))
    if not files:
        print(f"❌ No files in {CORPUS_DIR}")
        return

    print(f"Scanning {len(files)} papers...")

    all_tables = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        paper_id = doc.get("paper_id", path.stem)
        for tbl in doc.get("property_tables", []):
            score, reasons = score_table(tbl)
            all_tables.append({
                "paper_id":   paper_id,
                "table_id":   tbl.get("table_id", ""),
                "score":      score,
                "reasons":    reasons,
                "n_rows":     len(tbl.get("rows", [])),
                "n_cols":     len(tbl.get("header", [])),
                "caption":    tbl.get("caption", "")[:120],
                "header_preview": [str(h)[:40] for h in tbl.get("header", [])[:4]],
                "table":      tbl,  # keep for --dump
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(all_tables)
    if n == 0:
        print("❌ No tables found.")
        return

    scores = [t["score"] for t in all_tables]
    mean = sum(scores) / n
    median = sorted(scores)[n // 2]

    buckets = Counter()
    for s in scores:
        if s >= 80:   buckets["good (80-100)"] += 1
        elif s >= 60: buckets["fair (60-79)"] += 1
        elif s >= 40: buckets["poor (40-59)"] += 1
        else:         buckets["broken (0-39)"] += 1

    print()
    print("=" * 72)
    print(f"  TABLE AUDIT — {n} tables across {len(files)} papers")
    print("=" * 72)
    print(f"  Mean score:   {mean:.1f}/100")
    print(f"  Median score: {median}/100")
    print()
    print("  Distribution:")
    for label in ("good (80-100)", "fair (60-79)", "poor (40-59)", "broken (0-39)"):
        print(f"    {label:<18} {buckets[label]:>4}  ({buckets[label]/n*100:.1f}%)")

    # ── Failure reason counts ─────────────────────────────────────────────────
    reason_counts = Counter()
    for t in all_tables:
        for r in t["reasons"]:
            reason_counts[r] += 1

    print()
    print("  Top failure reasons:")
    for reason, count in reason_counts.most_common(10):
        print(f"    {reason:<35} {count:>4}")

    # ── Worst tables ──────────────────────────────────────────────────────────
    all_tables.sort(key=lambda t: t["score"])
    print()
    print("=" * 72)
    print("  WORST 20 TABLES")
    print("=" * 72)
    for i, t in enumerate(all_tables[:20], 1):
        print(f"  [{i:>2}] score={t['score']:>3}  {t['table_id']}")
        print(f"       rows={t['n_rows']}  cols={t['n_cols']}")
        print(f"       header_preview: {t['header_preview']}")
        print(f"       reasons: {', '.join(t['reasons']) or 'none'}")
        print()

    # ── Optional dump ─────────────────────────────────────────────────────────
    if dump_n > 0:
        print("=" * 72)
        print(f"  FULL DUMP — worst {dump_n} tables")
        print("=" * 72)
        for t in all_tables[:dump_n]:
            tbl = t["table"]
            print(f"\n--- {t['table_id']} (score={t['score']}) ---")
            print(f"  paper_id: {t['paper_id']}")
            print(f"  caption:  {t['caption']}")
            print(f"  header:   {tbl.get('header')}")
            for row in tbl.get("rows", [])[:3]:
                print(f"  row:      {row}")
            print(f"  text_repr (first 300 chars): {tbl.get('text_representation','')[:300]}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=" * 72)
    good_pct = buckets["good (80-100)"] / n * 100
    broken_pct = buckets["broken (0-39)"] / n * 100
    if good_pct >= 60 and broken_pct <= 10:
        print("  ✅ Most tables are usable. Table-aware retrieval should be fine.")
    elif broken_pct >= 30:
        print("  ❌ Too many broken tables. Fix extraction before claiming table benefits.")
    else:
        print("  ⚠️  Mixed quality. Table-aware retrieval helps on good tables, hurts on broken ones.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0,
                    help="Dump N worst tables in full for inspection")
    args = ap.parse_args()
    audit_corpus(dump_n=args.dump)