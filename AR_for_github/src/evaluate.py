"""
Module 5: Full evaluation pipeline
- RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
- Retrieval metrics: precision@k, recall@k
- Baseline comparison: RAG vs bare LLM
- LLM judge: Gemini flash lite (all API settings from .env)
- Embeddings: local all-MiniLM-L6-v2
"""

import json
import logging
import os
import sys
import csv
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from sentence_transformers import SentenceTransformer

from rag_pipeline import AlloyRAGPipeline, bare_llm_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ── RunConfig ──────────────────────────────────────────────────────────────────
run_config = RunConfig(
    max_workers=1,
    max_retries=5,
    max_wait=60,
    timeout=300,
)

# ── Rate limiter ───────────────────────────────────────────────────────────────
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.12,
    check_every_n_seconds=0.1,
    max_bucket_size=1,          
)

# ── RAGAS LLM judge ────────────────────────────────────────────────────────────
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=os.getenv("MODEL", "gemini-3.5-flash-lite"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv(
            "BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
        temperature=0.0,
        max_tokens=5000,
        max_retries=5,
        request_timeout=300.0,
        rate_limiter=rate_limiter,
        default_headers={
            "HTTP-Referer": "https://github.com/alloyrag",
            "X-Title":      "AlloyRAG",
        },
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# MINILM EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

class MiniLMEmbeddings(BaseRagasEmbeddings):
    def __init__(self, device: str = "cuda"):
        self._model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    query_id:                str
    query:                   str
    alloy_system:            str
    rag_answer:              str
    bare_llm_answer:         str
    retrieved_chunk_ids:     list
    retrieved_alloy_systems: list
    n_table_chunks:          int
    faithfulness_score:      Optional[float]
    answer_relevancy_score:  Optional[float]
    context_precision_score: Optional[float]
    context_recall_score:    Optional[float]
    expert_score_rag:        Optional[float]
    expert_score_bare:       Optional[float]
    difficulty:              str = ""
    notes:                   str = ""


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_retrieval_precision_recall(
    retrieved_ids:      list[str],
    relevant_paper_ids: list[str],
    k:                  int,
) -> tuple[float, float]:
    if not relevant_paper_ids:
        return 0.0, 0.0

    top_k = retrieved_ids[:k]
    retrieved_papers = set()

    for cid in top_k:
        if "_chunk_" in cid:
            retrieved_papers.add(cid.rsplit("_chunk_", 1)[0])
        elif "_table_" in cid:
            retrieved_papers.add(cid.rsplit("_table_", 1)[0])
        elif "_p" in cid:
            retrieved_papers.add(cid.split("_p")[0])
        else:
            retrieved_papers.add(cid)

    relevant_set = set(relevant_paper_ids)
    hits         = len(retrieved_papers & relevant_set)
    precision    = hits / k if k > 0 else 0.0
    recall       = hits / len(relevant_set) if relevant_set else 0.0
    return precision, recall


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _safe_aggregate(val) -> float:
    """Extract scalar from RAGAS result robustly."""
    if isinstance(val, (int, float)) and val == val:
        return round(float(val), 4)
    if isinstance(val, list):
        valid = [
            v for v in val
            if v is not None and isinstance(v, (int, float)) and v == v
        ]
        return round(sum(valid) / len(valid), 4) if valid else 0.0
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return 0.0


def _safe_list(df, col_name: str) -> list:
    """Extract per-query scores from RAGAS result dataframe."""
    if col_name not in df.columns:
        return []
    return [
        float(v) if v is not None and v == v else None
        for v in df[col_name].tolist()
    ]


def run_ragas_evaluation(
    queries:              list[str],
    rag_answers:          list[str],
    contexts:             list[list[str]],
    ground_truths:        list[str],
    corpus_ground_truths: list[str] = None,
    label:                str       = "AlloyRAG",
) -> dict:
    """
    Run RAGAS evaluation.

    WHAT IS COMPARED TO WHAT (per RAGAS 0.2.x docs):
    ─────────────────────────────────────────────────
    Faithfulness:
      Input:  answer (generated), contexts (retrieved chunks)
      Method: LLM checks if each claim in answer is supported by contexts
      Ground truth: NOT USED
      Measures: hallucination — does answer stay within retrieved evidence?

    AnswerRelevancy:
      Input:  question (user query), answer (generated)
      Method: embedding cosine similarity between question and answer
      Ground truth: NOT USED
      Measures: does the answer address what was asked?

    ContextPrecision:
      Input:  question, contexts, ground_truth
      Method: LLM ranks each context chunk by usefulness for the question
              given the ground truth as reference
      Ground truth: USED — as reference for what a good answer looks like
      Measures: are retrieved chunks ranked usefully?

    ContextRecall:
      Input:  contexts, ground_truth
      Method: LLM checks which statements in ground_truth are covered
              by the retrieved contexts
      Ground truth: USED — as source of statements to check coverage for
      Measures: does retrieval cover all needed information?

    SPLIT MODE RATIONALE:
      When corpus_ground_truths is provided, ContextPrecision and
      ContextRecall use corpus-grounded answers (written from actual
      corpus content) instead of expert answers (which contain
      knowledge beyond corpus → precision=0, recall=0.07).
      Faithfulness and AnswerRelevancy use expert ground_truth always
      (they actually ignore it, but kept for dataset completeness).
    """
    logger.info(f"[{label}] Initializing MiniLM embeddings...")
    ragas_embeddings = MiniLMEmbeddings(device="cuda")
    logger.info(f"[{label}] Embeddings ready")

    faithfulness_metric      = Faithfulness(llm=ragas_llm)
    answer_relevancy_metric  = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    context_precision_metric = ContextPrecision(llm=ragas_llm)
    context_recall_metric    = ContextRecall(llm=ragas_llm)

    safe_contexts = [
        c if (c and any(t.strip() for t in c))
        else ["No context retrieved."]
        for c in contexts
    ]

    use_split = (
        corpus_ground_truths is not None
        and len(corpus_ground_truths) == len(queries)
        and any(g.strip() for g in corpus_ground_truths)
    )

    if use_split:
        logger.info(
            f"[{label}] Split mode: "
            "faithfulness+relevancy vs expert GT, "
            "precision+recall vs corpus GT"
        )

        # Run 1: faithfulness + relevancy (expert ground truth)
        logger.info(f"[{label}] RAGAS 1/2: faithfulness + answer_relevancy...")
        ds1 = Dataset.from_dict({
            "question":     queries,
            "answer":       rag_answers,
            "contexts":     safe_contexts,
            "ground_truth": ground_truths,
        })
        r1 = ragas_evaluate(
            ds1,
            metrics=[faithfulness_metric, answer_relevancy_metric],
            run_config=run_config,
            raise_exceptions=False,
        )

        # Run 2: context precision + recall (corpus ground truth)
        logger.info(f"[{label}] RAGAS 2/2: context precision + recall...")
        ds2 = Dataset.from_dict({
            "question":     queries,
            "answer":       rag_answers,
            "contexts":     safe_contexts,
            "ground_truth": corpus_ground_truths,
        })
        r2 = ragas_evaluate(
            ds2,
            metrics=[context_precision_metric, context_recall_metric],
            run_config=run_config,
            raise_exceptions=False,
        )

        # FIX: No aliasing — each variable assigned separately
        per_q_faith  = []
        per_q_relev  = []
        per_q_prec   = []
        per_q_recall = []

        try:
            df1 = r1.to_pandas()
            df2 = r2.to_pandas()
            per_q_faith  = _safe_list(df1, "faithfulness")
            per_q_relev  = _safe_list(df1, "answer_relevancy")
            per_q_prec   = _safe_list(df2, "context_precision")
            per_q_recall = _safe_list(df2, "context_recall")

            unique = set(round(v, 3) for v in per_q_faith if v is not None)
            if len(unique) <= 1:
                logger.warning(f"[{label}] Faithfulness scores all identical — possible extraction issue")
            else:
                logger.info(f"[{label}] Per-query faithfulness varies ({len(unique)} unique values)")
        except Exception as e:
            logger.warning(f"[{label}] Per-query extraction failed: {e}")

        return {
            "faithfulness":           _safe_aggregate(r1["faithfulness"]),
            "answer_relevancy":       _safe_aggregate(r1["answer_relevancy"]),
            "context_precision":      _safe_aggregate(r2["context_precision"]),
            "context_recall":         _safe_aggregate(r2["context_recall"]),
            "per_query_faithfulness": per_q_faith,
            "per_query_relevancy":    per_q_relev,
            "per_query_precision":    per_q_prec,
            "per_query_recall":       per_q_recall,
            "ground_truth_mode":      "split",
        }

    else:
        # Single mode: all four metrics, expert ground truth
        logger.info(f"[{label}] RAGAS single run: all four metrics...")
        ds = Dataset.from_dict({
            "question":     queries,
            "answer":       rag_answers,
            "contexts":     safe_contexts,
            "ground_truth": ground_truths,
        })
        r = ragas_evaluate(
            ds,
            metrics=[
                faithfulness_metric,
                answer_relevancy_metric,
                context_precision_metric,
                context_recall_metric,
            ],
            run_config=run_config,
            raise_exceptions=False,
        )

        # FIX: No aliasing — each variable assigned separately
        per_q_faith  = []
        per_q_relev  = []
        per_q_prec   = []
        per_q_recall = []

        try:
            df = r.to_pandas()
            logger.info(f"[{label}] Dataframe columns: {list(df.columns)}")
            per_q_faith  = _safe_list(df, "faithfulness")
            per_q_relev  = _safe_list(df, "answer_relevancy")
            per_q_prec   = _safe_list(df, "context_precision")
            per_q_recall = _safe_list(df, "context_recall")

            unique = set(round(v, 3) for v in per_q_faith if v is not None)
            if len(unique) <= 1:
                logger.warning(f"[{label}] All faithfulness scores identical")
        except Exception as e:
            logger.warning(f"[{label}] Per-query extraction failed: {e}")

        return {
            "faithfulness":           _safe_aggregate(r["faithfulness"]),
            "answer_relevancy":       _safe_aggregate(r["answer_relevancy"]),
            "context_precision":      _safe_aggregate(r["context_precision"]),
            "context_recall":         _safe_aggregate(r["context_recall"]),
            "per_query_faithfulness": per_q_faith,
            "per_query_relevancy":    per_q_relev,
            "per_query_precision":    per_q_prec,
            "per_query_recall":       per_q_recall,
            "ground_truth_mode":      "single",
        }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FACTORY — shared indexes, single load, no double messages
# ══════════════════════════════════════════════════════════════════════════════

def _make_generation_pipeline(
    source: AlloyRAGPipeline,
    top_k:  int = 5,
) -> AlloyRAGPipeline:
    """
    Create a generation pipeline that shares all loaded indexes
    from an existing retrieval pipeline.
    Only top_k differs — no re-loading of embedder, ChromaDB, BM25, ID maps.
    Prevents double initialization log messages.
    """
    gen = object.__new__(AlloyRAGPipeline)

    # Copy all attributes from source
    gen.top_k       = top_k
    gen.dense_k     = source.dense_k
    gen.bm25_k      = source.bm25_k

    # Share loaded resources — no reload
    gen.embedder      = source.embedder
    gen.chroma_client = source.chroma_client
    gen.collection    = source.collection
    gen.bm25          = source.bm25
    gen.bm25_ids      = source.bm25_ids
    gen.bm25_texts    = source.bm25_texts
    gen.id_to_text    = source.id_to_text
    gen.id_to_meta    = source.id_to_meta

    return gen


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    gold_qa_path: str       = "data/gold_qa.json",
    output_dir:   str       = "data/evaluation",
    top_k_values: list[int] = None,
) -> None:
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    partial_path = Path(output_dir) / "partial_results.json"

    # ── Load gold QA ───────────────────────────────────────────────────────────
    if not Path(gold_qa_path).exists():
        logger.error(f"gold_qa.json not found at: {gold_qa_path}")
        return

    with open(gold_qa_path, encoding="utf-8") as f:
        gold_qa = json.load(f)
    logger.info(f"Loaded {len(gold_qa)} gold QA pairs")

    # Check corpus_grounded_answer availability
    n_corpus = sum(
        1 for q in gold_qa
        if q.get("corpus_grounded_answer", "").strip()
    )
    logger.info(f"corpus_grounded_answer filled: {n_corpus}/{len(gold_qa)}")
    use_split = n_corpus >= len(gold_qa) // 2

    if use_split:
        logger.info("Split mode active — context metrics use corpus-grounded answers")
    else:
        logger.info(
            "Single mode — fill corpus_grounded_answer in gold_qa.json "
            "to get meaningful context_precision and context_recall"
        )

    # ── Initialize pipelines — ONE load, shared indexes ────────────────────────
    logger.info("Loading AlloyRAG indexes (single load)...")
    retrieval_pipeline  = AlloyRAGPipeline(top_k=max(top_k_values))
    generation_pipeline = _make_generation_pipeline(retrieval_pipeline, top_k=5)
    logger.info("Both pipelines ready — indexes shared, no double loading")

    # ── Retrieval metrics ──────────────────────────────────────────────────────
    logger.info("Computing retrieval metrics...")
    retrieval_results = {k: {"precision": [], "recall": []} for k in top_k_values}

    for qa in gold_qa:
        relevant_ids = qa.get("relevant_paper_ids", [])
        if not relevant_ids:
            continue
        chunks        = retrieval_pipeline.retrieve(
            qa["query"],
            alloy_filter=qa.get("alloy_system"),
        )
        retrieved_ids = [c.chunk_id for c in chunks]
        for k in top_k_values:
            prec, rec = compute_retrieval_precision_recall(
                retrieved_ids, relevant_ids, k
            )
            retrieval_results[k]["precision"].append(prec)
            retrieval_results[k]["recall"].append(rec)

    avg_retrieval: dict[str, float] = {}
    for k, m in retrieval_results.items():
        avg_retrieval[f"precision@{k}"] = (
            round(sum(m["precision"]) / len(m["precision"]), 4)
            if m["precision"] else 0.0
        )
        avg_retrieval[f"recall@{k}"] = (
            round(sum(m["recall"]) / len(m["recall"]), 4)
            if m["recall"] else 0.0
        )
    logger.info(f"Retrieval metrics: {avg_retrieval}")

    # ── Initialize lists ───────────────────────────────────────────────────────
    queries_list:         list[str]        = []
    rag_answers:          list[str]        = []
    bare_answers:         list[str]        = []
    contexts_list:        list[list[str]]  = []
    ground_truths:        list[str]        = []
    corpus_ground_truths: list[str]        = []
    eval_records:         list[EvalResult] = []

    # ── Skip generation if complete partial results exist ──────────────────────
    already_done = False

    if partial_path.exists():
        try:
            with open(partial_path, encoding="utf-8") as f:
                partial = json.load(f)

            existing = partial.get("records_so_far", [])

            if len(existing) >= len(gold_qa):
                logger.info(
                    f"Found {len(existing)} completed answers — "
                    "skipping generation"
                )
                already_done = True
                gold_qa_map  = {q["query_id"]: q for q in gold_qa}
                valid_fields = set(EvalResult.__dataclass_fields__.keys())

                for r in existing:
                    qid = r.get("query_id", "")
                    qa  = gold_qa_map.get(qid, {})

                    queries_list.append(r.get("query", ""))
                    rag_answers.append(r.get("rag_answer", ""))
                    bare_answers.append(r.get("bare_llm_answer", ""))
                    ground_truths.append(qa.get("ground_truth_answer", ""))
                    corpus_ground_truths.append(
                        qa.get("corpus_grounded_answer", "")
                    )

                    chunk_ids   = r.get("retrieved_chunk_ids", [])
                    chunk_texts = [
                        generation_pipeline.id_to_text.get(cid, "")
                        for cid in chunk_ids
                        if generation_pipeline.id_to_text.get(cid, "").strip()
                    ]
                    if not chunk_texts:
                        chunk_texts = [r.get("rag_answer", "")]
                    contexts_list.append(chunk_texts)

                    fr = {k: v for k, v in r.items() if k in valid_fields}
                    fr.setdefault("difficulty", "")
                    fr.setdefault("notes", "")
                    fr.setdefault("faithfulness_score", None)
                    fr.setdefault("answer_relevancy_score", None)
                    fr.setdefault("context_precision_score", None)
                    fr.setdefault("context_recall_score", None)
                    fr.setdefault("expert_score_rag", None)
                    fr.setdefault("expert_score_bare", None)
                    eval_records.append(EvalResult(**fr))

        except Exception as e:
            logger.warning(f"Could not load partial results: {e} — regenerating")
            already_done         = False
            queries_list         = []
            rag_answers          = []
            bare_answers         = []
            contexts_list        = []
            ground_truths        = []
            corpus_ground_truths = []
            eval_records         = []

    # ── Generate answers ───────────────────────────────────────────────────────
    if not already_done:
        logger.info("Generating RAG and bare LLM answers...")

        for i, qa in enumerate(gold_qa):
            logger.info(
                f"[{i+1}/{len(gold_qa)}] {qa['query_id']}: "
                f"{qa['query'][:55]}..."
            )

            rag_result  = generation_pipeline.query(
                qa["query"],
                alloy_filter=qa.get("alloy_system"),
            )
            bare_answer = bare_llm_answer(qa["query"])

            retrieved_ids     = [c.chunk_id     for c in rag_result.retrieved_chunks]
            retrieved_systems = [c.alloy_system for c in rag_result.retrieved_chunks]
            n_tables          = sum(
                1 for c in rag_result.retrieved_chunks
                if c.chunk_type == "table"
            )
            chunk_texts = [c.text for c in rag_result.retrieved_chunks]

            queries_list.append(qa["query"])
            rag_answers.append(rag_result.answer)
            bare_answers.append(bare_answer)
            contexts_list.append(chunk_texts)
            ground_truths.append(qa["ground_truth_answer"])
            corpus_ground_truths.append(
                qa.get("corpus_grounded_answer", "")
            )

            eval_records.append(EvalResult(
                query_id=qa["query_id"],
                query=qa["query"],
                alloy_system=qa.get("alloy_system", "Mixed"),
                rag_answer=rag_result.answer,
                bare_llm_answer=bare_answer,
                retrieved_chunk_ids=retrieved_ids,
                retrieved_alloy_systems=retrieved_systems,
                n_table_chunks=n_tables,
                faithfulness_score=None,
                answer_relevancy_score=None,
                context_precision_score=None,
                context_recall_score=None,
                expert_score_rag=None,
                expert_score_bare=None,
                difficulty=qa.get("difficulty", ""),
            ))

            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed_queries": i + 1,
                        "total_queries":     len(gold_qa),
                        "records_so_far":    [asdict(r) for r in eval_records],
                    },
                    f, indent=2, ensure_ascii=False,
                )

        logger.info("All answers generated")

    # ── RAGAS: AlloyRAG ───────────────────────────────────────────────────────
    ragas_scores = run_ragas_evaluation(
        queries_list,
        rag_answers,
        contexts_list,
        ground_truths,
        corpus_ground_truths=corpus_ground_truths if use_split else None,
        label="AlloyRAG",
    )

    per_q_faith  = ragas_scores.pop("per_query_faithfulness", [])
    per_q_relev  = ragas_scores.pop("per_query_relevancy", [])
    per_q_prec   = ragas_scores.pop("per_query_precision", [])
    per_q_recall = ragas_scores.pop("per_query_recall", [])
    ragas_scores.pop("ground_truth_mode", None)

    logger.info(f"AlloyRAG RAGAS scores: {ragas_scores}")

    for i, record in enumerate(eval_records):
        record.faithfulness_score = (
            per_q_faith[i]
            if i < len(per_q_faith) and per_q_faith[i] is not None
            else ragas_scores.get("faithfulness")
        )
        record.answer_relevancy_score = (
            per_q_relev[i]
            if i < len(per_q_relev) and per_q_relev[i] is not None
            else ragas_scores.get("answer_relevancy")
        )
        record.context_precision_score = (
            per_q_prec[i]
            if i < len(per_q_prec) and per_q_prec[i] is not None
            else ragas_scores.get("context_precision")
        )
        record.context_recall_score = (
            per_q_recall[i]
            if i < len(per_q_recall) and per_q_recall[i] is not None
            else ragas_scores.get("context_recall")
        )

    # ── RAGAS: Bare LLM — always single mode (saves time, only needs faith) ───
    logger.info("Running RAGAS on bare LLM answers...")
    try:
        bare_ragas = run_ragas_evaluation(
            queries_list,
            bare_answers,
            contexts_list,
            ground_truths,
            corpus_ground_truths=None,   # bare LLM always single mode
            label="BareALLM",
        )
        bare_ragas.pop("per_query_faithfulness", None)
        bare_ragas.pop("per_query_relevancy", None)
        bare_ragas.pop("per_query_precision", None)
        bare_ragas.pop("per_query_recall", None)
        bare_ragas.pop("ground_truth_mode", None)
        logger.info(f"Bare LLM RAGAS scores: {bare_ragas}")
    except Exception as e:
        logger.warning(f"Bare LLM RAGAS failed: {e}")
        bare_ragas = {}

    # ── Save JSON ──────────────────────────────────────────────────────────────
    results_dict = {
        "retrieval_metrics":      avg_retrieval,
        "ragas_aggregate":        ragas_scores,
        "bare_llm_ragas":         bare_ragas,
        "ground_truth_mode_used": "split" if use_split else "single",
        "per_query_results":      [asdict(r) for r in eval_records],
    }

    out_path = Path(output_dir) / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {out_path}")

    # ── Save expert CSV ────────────────────────────────────────────────────────
    expert_path = Path(output_dir) / "expert_annotation_template.csv"
    with open(expert_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "alloy_system", "difficulty", "query",
            "rag_answer", "bare_llm_answer", "n_table_chunks_retrieved",
            "faithfulness_score", "answer_relevancy_score",
            "context_precision_score",
            "expert_score_rag_1_5", "expert_score_bare_1_5", "expert_notes",
        ])
        writer.writeheader()
        for r in eval_records:
            writer.writerow({
                "query_id":                r.query_id,
                "alloy_system":            r.alloy_system,
                "difficulty":              r.difficulty,
                "query":                   r.query,
                "rag_answer":              r.rag_answer[:800],
                "bare_llm_answer":         r.bare_llm_answer[:800],
                "n_table_chunks_retrieved": r.n_table_chunks,
                "faithfulness_score":      r.faithfulness_score   or "",
                "answer_relevancy_score":  r.answer_relevancy_score or "",
                "context_precision_score": r.context_precision_score or "",
                "expert_score_rag_1_5":    "",
                "expert_score_bare_1_5":   "",
                "expert_notes":            "",
            })
    logger.info(f"Expert CSV saved: {expert_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"\nGround truth mode: {'split (corpus-grounded)' if use_split else 'single (expert)'}", flush=True)

    print("\nRetrieval Metrics:", flush=True)
    for k, v in avg_retrieval.items():
        print(f"  {k}: {v:.4f}", flush=True)

    print("\nRAGAS — AlloyRAG:", flush=True)
    for metric, score in ragas_scores.items():
        print(f"  {metric}: {score:.4f}", flush=True)

    if bare_ragas:
        print("\nRAGAS — Bare LLM baseline:", flush=True)
        for metric, score in bare_ragas.items():
            print(f"  {metric}: {score:.4f}", flush=True)

        rag_faith  = ragas_scores.get("faithfulness", 0)
        bare_faith = bare_ragas.get("faithfulness", 0)
        if bare_faith > 0:
            pct = (rag_faith - bare_faith) / bare_faith * 100
            print(
                f"\nFaithfulness improvement: "
                f"{rag_faith:.4f} vs {bare_faith:.4f} (+{pct:.1f}%)",
                flush=True,
            )

    print("\nPer-Query Results:", flush=True)
    for r in eval_records:
        faith = f"{r.faithfulness_score:.3f}" if r.faithfulness_score is not None else "N/A"
        relev = f"{r.answer_relevancy_score:.3f}" if r.answer_relevancy_score is not None else "N/A"
        prec  = f"{r.context_precision_score:.3f}" if r.context_precision_score is not None else "N/A"
        print(
            f"  {r.query_id} [{r.difficulty:<6}] "
            f"tables={r.n_table_chunks} | "
            f"faith={faith} | relev={relev} | ctx_prec={prec}",
            flush=True,
        )

    print(f"\nSaved to: {output_dir}/", flush=True)
    print("Next: fill expert scores in expert_annotation_template.csv", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_full_evaluation()