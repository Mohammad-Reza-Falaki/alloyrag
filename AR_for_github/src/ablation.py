"""
Module 6: Ablation experiments (checkpointed)

Ablations:
  1. k values       (1, 3, 5, 10)
  2. retrieval mode (dense-only vs BM25-only vs hybrid RRF)
  3. chunk content  (text-only vs text + tables) — matched narrative context

Checkpointing:
  - Every generated answer is saved to ablation_checkpoint.json immediately
  - Every RAGAS score is saved to ablation_checkpoint.json immediately
  - Resume support: skipped queries on re-run
  - Aggregates also written to ablation_results.json at the end

Usage:
    python src/ablation.py --only chunk
    python src/ablation.py --only chunk --queries Q001 Q003 Q006 Q012 Q017 --label tablerelevant
    python src/ablation.py --only k
    python src/ablation.py --only retrieval
    python src/ablation.py --only all
    python src/ablation.py --only chunk --reset       # wipe checkpoint and start fresh
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from sentence_transformers import SentenceTransformer

from rag_pipeline import AlloyRAGPipeline, RetrievedChunk


# ── Configuration ─────────────────────────────────────────────────────────────

GOLD_QA_PATH       = "data/gold_qa.json"
OUTPUT_DIR         = "data/evaluation"
OUT_PATH           = Path(OUTPUT_DIR) / "ablation_results.json"
CHECKPOINT_PATH    = Path(OUTPUT_DIR) / "ablation_checkpoint.json"

N_STRATIFIED   = 7
N_TEXT_CHUNKS  = 5      # narrative chunks (identical in both chunk-ablation conditions)
N_TABLE_CHUNKS = 2      # table chunks appended only in text_and_table

RAGAS_MAX_TOKENS = 8000
RAGAS_MAX_WAIT   = 600
SLEEP_BETWEEN    = 60

ABLATION_KEYS = {
    "k":         "k_ablation",
    "retrieval": "retrieval_mode_ablation",
    "chunk":     "chunk_content_ablation",
}


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# ── Rate limiter + RAGAS judge ────────────────────────────────────────────────

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=os.getenv("MODEL", "gemini-3.5-flash-lite"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv(
            "BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        temperature=0.0,
        max_tokens=RAGAS_MAX_TOKENS,
        rate_limiter=rate_limiter,
        default_headers={
            "HTTP-Referer": "https://github.com/alloyrag",
            "X-Title":      "AlloyRAG",
        },
    )
)

RAGAS_RUN_CONFIG = RunConfig(
    max_workers=1,
    max_wait=RAGAS_MAX_WAIT,
)


# ── Embeddings (loaded once) ──────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"RAGAS embeddings device: {DEVICE}")


class MiniLMEmbeddings(BaseRagasEmbeddings):
    def __init__(self, device: str = DEVICE):
        self._model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            [text], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


_RAGAS_EMBEDDINGS = MiniLMEmbeddings(device=DEVICE)


# ── Checkpoint ────────────────────────────────────────────────────────────────
#
# Structure (single JSON file):
#
# {
#   "k_ablation": {
#     "k=5": {
#       "Q001": {
#         "generated": true,
#         "answer": "...",
#         "contexts": ["...", "..."],
#         "chunk_ids": ["AX_...", "..."],
#         "chunk_types": ["text", "table", ...],
#         "ragas_scored": true,
#         "faithfulness": 0.87,
#         "answer_relevancy": 0.91
#       },
#       ...
#     },
#     ...
#   },
#   "chunk_content_ablation": {
#     "text_only": {
#       "Q001": { ... },
#       ...
#     },
#     ...
#   }
# }

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Checkpoint unreadable ({e}) — starting fresh")
        return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    # ── Nested accessors ─────────────────────────────────────────────────
    def _get_node(self, *keys, create: bool = False) -> dict | None:
        cur = self.data
        for k in keys:
            if not isinstance(cur, dict):
                return None
            if k not in cur:
                if not create:
                    return None
                cur[k] = {}
            cur = cur[k]
        return cur

    def get_query(self, ablation: str, condition: str, qid: str) -> dict:
        node = self._get_node(ablation, condition, create=True)
        return node.get(qid, {}) if node else {}

    def has_generation(self, ablation: str, condition: str, qid: str) -> bool:
        return self.get_query(ablation, condition, qid).get("generated", False)

    def has_scores(self, ablation: str, condition: str, qid: str) -> bool:
        return self.get_query(ablation, condition, qid).get("ragas_scored", False)

    def save_generation(
        self,
        ablation: str,
        condition: str,
        qid: str,
        answer: str,
        contexts: list[str],
        chunk_ids: list[str],
        chunk_types: list[str],
    ) -> None:
        node = self._get_node(ablation, condition, create=True)
        entry = node.get(qid, {})
        entry.update({
            "generated":   True,
            "answer":      answer,
            "contexts":    contexts,
            "chunk_ids":   chunk_ids,
            "chunk_types": chunk_types,
        })
        node[qid] = entry
        self.save()

    def save_scores(
        self,
        ablation: str,
        condition: str,
        qid: str,
        faith: float | None,
        relev: float | None,
    ) -> None:
        node = self._get_node(ablation, condition, create=True)
        entry = node.get(qid, {})
        entry.update({
            "ragas_scored":     True,
            "faithfulness":     faith,
            "answer_relevancy": relev,
        })
        node[qid] = entry
        self.save()

    def list_queries(self, ablation: str, condition: str) -> list[str]:
        node = self._get_node(ablation, condition)
        return list(node.keys()) if node else []


# ── Utilities ─────────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    clean = [s for s in scores
             if s is not None and isinstance(s, (int, float)) and s == s]
    if len(clean) < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(n_boot):
        sample = [clean[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (
        round(means[int(n_boot * alpha / 2)], 4),
        round(means[int(n_boot * (1 - alpha / 2))], 4),
    )


def stratified_sample(
    gold_qa: list[dict],
    n: int = N_STRATIFIED,
    target_dist: dict | None = None,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    if target_dist is None:
        target_dist = {"easy": 3, "medium": 2, "hard": 2}

    by_diff: dict[str, list[dict]] = defaultdict(list)
    for qa in gold_qa:
        by_diff[qa.get("difficulty", "unknown")].append(qa)

    selected: list[dict] = []
    used_alloys: set[str] = set()

    for diff, count in target_dist.items():
        pool = list(by_diff.get(diff, []))
        pool.sort(key=lambda qa: (qa.get("alloy_system") in used_alloys, rng.random()))
        for qa in pool:
            if count <= 0 or len(selected) >= n:
                break
            selected.append(qa)
            used_alloys.add(qa.get("alloy_system", "Unknown"))
            count -= 1

    if len(selected) < n:
        for qa in gold_qa:
            if len(selected) >= n:
                break
            if qa not in selected:
                selected.append(qa)

    return selected[:n]


def select_queries(gold_qa: list[dict], query_ids: list[str] | None) -> list[dict]:
    if query_ids:
        id_map = {qa["query_id"]: qa for qa in gold_qa}
        missing = [qid for qid in query_ids if qid not in id_map]
        if missing:
            raise ValueError(f"Unknown query IDs: {missing}")
        return [id_map[qid] for qid in query_ids]
    return stratified_sample(gold_qa, n=N_STRATIFIED, seed=42)


def log_subset(ablation_qa: list[dict]) -> None:
    logger.info(f"Running on {len(ablation_qa)} queries:")
    for qa in ablation_qa:
        logger.info(
            f"  {qa['query_id']} [{qa.get('difficulty','?'):<6}] "
            f"[{qa.get('alloy_system','?'):<6}] {qa['query'][:60]}"
        )


# ── RAGAS scoring (per query, so it's resumable) ──────────────────────────────

def score_one(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> tuple[float | None, float | None]:
    """
    Score a single query. Returns (faithfulness, answer_relevancy); either may
    be None if the metric failed.
    """
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=_RAGAS_EMBEDDINGS),
    ]

    safe_ctx = contexts if (contexts and any(t.strip() for t in contexts)) \
               else ["No context retrieved."]

    dataset = Dataset.from_dict({
        "question":     [query],
        "answer":       [answer],
        "contexts":     [safe_ctx],
        "ground_truth": [ground_truth],
    })

    try:
        result = ragas_evaluate(
            dataset,
            metrics=metrics,
            run_config=RAGAS_RUN_CONFIG,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        faith = relev = None
        if "faithfulness" in df.columns:
            v = df["faithfulness"].iloc[0]
            faith = float(v) if v is not None and v == v else None
        if "answer_relevancy" in df.columns:
            v = df["answer_relevancy"].iloc[0]
            relev = float(v) if v is not None and v == v else None
        return faith, relev
    except Exception as e:
        logger.warning(f"RAGAS failed for one query: {e}")
        return None, None


# ── Generic runner for one condition ──────────────────────────────────────────

def generate_and_score(
    ablation: str,
    condition: str,
    gold_qa: list[dict],
    pipeline: AlloyRAGPipeline,
    checkpoint: Checkpoint,
    answer_fn,
) -> None:
    """
    For each query: generate the answer (if not already done), then score it
    (if not already done). Both steps save to the checkpoint immediately.

    `answer_fn(pipeline, qa)` must return (answer, chunks, contexts).
    """
    total = len(gold_qa)
    for i, qa in enumerate(gold_qa, 1):
        qid = qa["query_id"]
        logger.info(f"[{ablation}/{condition}] {i}/{total} {qid}")

        # ── Generate ─────────────────────────────────────────────────────
        if not checkpoint.has_generation(ablation, condition, qid):
            try:
                answer, chunks, contexts = answer_fn(pipeline, qa)
                chunk_ids   = [c.chunk_id    for c in chunks]
                chunk_types = [c.chunk_type  for c in chunks]
                checkpoint.save_generation(
                    ablation, condition, qid,
                    answer=answer,
                    contexts=contexts,
                    chunk_ids=chunk_ids,
                    chunk_types=chunk_types,
                )
                logger.info(
                    f"    generated ({len(answer)} chars, {len(chunks)} chunks)"
                )
            except Exception as e:
                logger.error(f"    generation failed: {e}")
                continue
        else:
            logger.info("    generation already saved — skipping")

        # ── Score ────────────────────────────────────────────────────────
        if not checkpoint.has_scores(ablation, condition, qid):
            entry = checkpoint.get_query(ablation, condition, qid)
            faith, relev = score_one(
                query=qa["query"],
                answer=entry["answer"],
                contexts=entry["contexts"],
                ground_truth=qa["ground_truth_answer"],
            )
            checkpoint.save_scores(ablation, condition, qid, faith, relev)
            logger.info(f"    scored faith={faith} relev={relev}")
        else:
            logger.info("    scores already saved — skipping")


def aggregate_condition(
    ablation: str,
    condition: str,
    gold_qa: list[dict],
    checkpoint: Checkpoint,
) -> dict:
    """Compute aggregate stats from per-query checkpoint entries."""
    faiths, relevs = [], []
    n_tables_per_query = []

    for qa in gold_qa:
        entry = checkpoint.get_query(ablation, condition, qa["query_id"])
        if entry.get("faithfulness") is not None:
            faiths.append(entry["faithfulness"])
        if entry.get("answer_relevancy") is not None:
            relevs.append(entry["answer_relevancy"])
        chunk_types = entry.get("chunk_types", [])
        n_tables_per_query.append(sum(1 for t in chunk_types if t == "table"))

    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    out = {
        "faithfulness":     _mean(faiths),
        "answer_relevancy": _mean(relevs),
        "n_faith":          len(faiths),
        "n_relev":          len(relevs),
        "n_queries":        len(gold_qa),
        "faith_ci":         list(bootstrap_ci(faiths)),
        "relev_ci":         list(bootstrap_ci(relevs)),
    }
    if n_tables_per_query and any(n_tables_per_query):
        out["n_table_per_query_mean"] = round(
            sum(n_tables_per_query) / len(n_tables_per_query), 2
        )
    return out


# ── Ablation-specific answer_fn builders ──────────────────────────────────────

def make_answer_fn_k():
    def _fn(pipeline, qa):
        rag = pipeline.query(qa["query"], alloy_filter=qa.get("alloy_system"))
        return rag.answer, rag.retrieved_chunks, [c.text for c in rag.retrieved_chunks]
    return _fn


def _make_chunk_objs(pipeline, raw, source):
    objs = []
    for cid, score in raw:
        text = pipeline.id_to_text.get(cid, "")
        meta = pipeline.id_to_meta.get(cid, {})
        if not text:
            continue
        objs.append(RetrievedChunk(
            chunk_id=cid, text=text,
            paper_id=meta.get("paper_id", ""),
            alloy_system=meta.get("alloy_system", ""),
            chunk_type=meta.get("chunk_type", "text"),
            section=meta.get("section", ""),
            score=score, rank_source=source,
        ))
    return objs


def make_answer_fn_retrieval(mode: str):
    def _fn(pipeline, qa):
        alloy_filter = qa.get("alloy_system")
        if mode == "dense_only":
            raw = pipeline._dense_retrieve(qa["query"], k=5, alloy_filter=alloy_filter)
            chunks = _make_chunk_objs(pipeline, raw[:5], "dense")
        elif mode == "bm25_only":
            raw = pipeline._bm25_retrieve(qa["query"], k=5, alloy_filter=alloy_filter)
            chunks = _make_chunk_objs(pipeline, raw[:5], "bm25")
        else:  # hybrid
            rag = pipeline.query(qa["query"], alloy_filter=alloy_filter)
            chunks = rag.retrieved_chunks
        answer = pipeline.generate(qa["query"], chunks)
        return answer, chunks, [c.text for c in chunks]
    return _fn


def make_answer_fn_chunk(mode: str, n_text: int, n_table: int):
    """mode = 'text_only' or 'text_and_table'."""
    def _fn(pipeline, qa):
        alloy_filter = qa.get("alloy_system")

        original_top_k = pipeline.top_k
        try:
            pipeline.top_k = n_text
            text_chunks = pipeline.retrieve(
                qa["query"],
                alloy_filter=alloy_filter,
                chunk_type_filter="text",
            )

            if mode == "text_only":
                filtered = text_chunks
            else:
                pipeline.top_k = n_table
                table_chunks = pipeline.retrieve(
                    qa["query"],
                    alloy_filter=alloy_filter,
                    chunk_type_filter="table",
                )
                filtered = text_chunks + table_chunks
        finally:
            pipeline.top_k = original_top_k

        answer = pipeline.generate(qa["query"], filtered)
        return answer, filtered, [c.text for c in filtered]
    return _fn


# ── Ablation drivers ──────────────────────────────────────────────────────────

def run_k_ablation(
    gold_qa: list[dict],
    pipeline: AlloyRAGPipeline,
    checkpoint: Checkpoint,
    k_values: list[int] | None = None,
) -> dict:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    ablation = ABLATION_KEYS["k"]
    answer_fn = make_answer_fn_k()
    results = {}

    for k in k_values:
        condition = f"k={k}"
        logger.info("=" * 60)
        logger.info(f"ABLATION k — condition {condition}")
        logger.info("=" * 60)
        pipeline.top_k = k
        generate_and_score(ablation, condition, gold_qa, pipeline, checkpoint, answer_fn)
        results[condition] = aggregate_condition(ablation, condition, gold_qa, checkpoint)
        _log_scores(condition, results[condition])

    return results


def run_retrieval_mode_ablation(
    gold_qa: list[dict],
    pipeline: AlloyRAGPipeline,
    checkpoint: Checkpoint,
) -> dict:
    ablation = ABLATION_KEYS["retrieval"]
    results = {}

    for mode in ["dense_only", "bm25_only", "hybrid"]:
        logger.info("=" * 60)
        logger.info(f"ABLATION retrieval — condition {mode}")
        logger.info("=" * 60)
        answer_fn = make_answer_fn_retrieval(mode)
        generate_and_score(ablation, mode, gold_qa, pipeline, checkpoint, answer_fn)
        results[mode] = aggregate_condition(ablation, mode, gold_qa, checkpoint)
        _log_scores(mode, results[mode])

    return results


def run_chunk_content_ablation(
    gold_qa: list[dict],
    pipeline: AlloyRAGPipeline,
    checkpoint: Checkpoint,
    n_text: int = N_TEXT_CHUNKS,
    n_table: int = N_TABLE_CHUNKS,
) -> dict:
    ablation = ABLATION_KEYS["chunk"]
    results = {}

    for mode in ["text_only", "text_and_table"]:
        logger.info("=" * 60)
        logger.info(f"ABLATION chunk — condition {mode} "
                    f"(n_text={n_text}, n_table={n_table})")
        logger.info("=" * 60)
        answer_fn = make_answer_fn_chunk(mode, n_text, n_table)
        generate_and_score(ablation, mode, gold_qa, pipeline, checkpoint, answer_fn)
        results[mode] = aggregate_condition(ablation, mode, gold_qa, checkpoint)
        _log_scores(mode, results[mode])

    return results


# ── Logging helper ────────────────────────────────────────────────────────────

def _log_scores(name: str, scores: dict) -> None:
    f_ci = scores.get("faith_ci", [0, 0])
    r_ci = scores.get("relev_ci", [0, 0])
    extra = ""
    if "n_table_per_query_mean" in scores:
        extra = f" tables/query={scores['n_table_per_query_mean']:.2f}"
    print(
        f"  {name:<20} "
        f"faith={scores['faithfulness']:.3f} "
        f"[{f_ci[0]:.3f},{f_ci[1]:.3f}] (n={scores['n_faith']}) | "
        f"relev={scores['answer_relevancy']:.3f} "
        f"[{r_ci[0]:.3f},{r_ci[1]:.3f}] (n={scores['n_relev']})"
        f"{extra}",
        flush=True,
    )


# ── Driver ────────────────────────────────────────────────────────────────────

def _load_results_json() -> dict:
    if OUT_PATH.exists():
        try:
            with open(OUT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_results_json(results: dict) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Aggregates saved to {OUT_PATH}")


def run_single_ablation(
    which: str,
    query_ids: list[str] | None = None,
    label: str | None = None,
) -> None:
    with open(GOLD_QA_PATH, encoding="utf-8") as f:
        gold_qa = json.load(f)

    ablation_qa = select_queries(gold_qa, query_ids)
    log_subset(ablation_qa)

    checkpoint = Checkpoint(CHECKPOINT_PATH)
    pipeline = AlloyRAGPipeline(top_k=5, dense_k=15, bm25_k=15)

    if which == "k":
        key = ABLATION_KEYS["k"]
        results = run_k_ablation(ablation_qa, pipeline, checkpoint)
    elif which == "retrieval":
        key = ABLATION_KEYS["retrieval"]
        results = run_retrieval_mode_ablation(ablation_qa, pipeline, checkpoint)
    elif which == "chunk":
        base_key = ABLATION_KEYS["chunk"]
        key = f"{base_key}_{label}" if label else base_key
        results = run_chunk_content_ablation(ablation_qa, pipeline, checkpoint)
    else:
        raise ValueError(f"Unknown ablation: {which}")

    # Merge into ablation_results.json (preserving existing entries)
    final = _load_results_json()
    final[key] = results
    final[f"{key}_query_ids"] = [qa["query_id"] for qa in ablation_qa]
    _save_results_json(final)

    _print_summary(key, results)


def run_all_ablations(query_ids: list[str] | None = None) -> None:
    with open(GOLD_QA_PATH, encoding="utf-8") as f:
        gold_qa = json.load(f)

    ablation_qa = select_queries(gold_qa, query_ids)
    log_subset(ablation_qa)

    checkpoint = Checkpoint(CHECKPOINT_PATH)
    pipeline = AlloyRAGPipeline(top_k=5, dense_k=15, bm25_k=15)

    final = _load_results_json()
    final["stratified_query_ids"] = [qa["query_id"] for qa in ablation_qa]

    for which in ["k", "retrieval", "chunk"]:
        logger.info("=" * 72)
        logger.info(f"ABLATION: {which}")
        logger.info("=" * 72)

        if which == "k":
            key = ABLATION_KEYS["k"]
            results = run_k_ablation(ablation_qa, pipeline, checkpoint)
        elif which == "retrieval":
            key = ABLATION_KEYS["retrieval"]
            results = run_retrieval_mode_ablation(ablation_qa, pipeline, checkpoint)
        else:
            key = ABLATION_KEYS["chunk"]
            results = run_chunk_content_ablation(ablation_qa, pipeline, checkpoint)

        final[key] = results
        final[f"{key}_query_ids"] = [qa["query_id"] for qa in ablation_qa]
        _save_results_json(final)
        _print_summary(key, results)

        time.sleep(SLEEP_BETWEEN)


def _print_summary(key: str, results: dict) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"ABLATION SUMMARY — {key}", flush=True)
    print("=" * 72, flush=True)
    for name, scores in results.items():
        _log_scores(name, scores)
    print("=" * 72, flush=True)


def _reset_checkpoint() -> None:
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info(f"Checkpoint wiped: {CHECKPOINT_PATH}")
    if OUT_PATH.exists():
        OUT_PATH.unlink()
        logger.info(f"Results wiped: {OUT_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", type=str, default="chunk",
        choices=["all", "k", "retrieval", "chunk"],
    )
    ap.add_argument(
        "--queries", type=str, nargs="+", default=None,
        help="Optional query-ID subset (e.g. --queries Q001 Q003 Q006)",
    )
    ap.add_argument(
        "--label", type=str, default=None,
        help="Suffix appended to the output key (chunk only)",
    )
    ap.add_argument(
        "--reset", action="store_true",
        help="Wipe checkpoint and results, then run",
    )
    args = ap.parse_args()

    if args.reset:
        _reset_checkpoint()

    if args.only == "all":
        run_all_ablations(query_ids=args.queries)
    else:
        run_single_ablation(args.only, query_ids=args.queries, label=args.label)