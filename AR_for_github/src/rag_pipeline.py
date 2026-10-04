"""
Module 4: Core RAG pipeline
Hybrid retrieval (dense + BM25) with RRF reranking + LLM generation
Uses Gemini OpenAI-compatible endpoint
"""

import os
import pickle
import logging
import sys
import re
from typing import Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH  = "data/chroma_db"
BM25_INDEX_PATH = "data/bm25_index.pkl"
ID_MAP_PATH     = "data/id_to_text.pkl"
LLM_MODEL       = os.getenv("MODEL", "gemini-3.5-flash-lite")

# ── LLM Client ─────────────────────────────────────────────────────────────────
llm_client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
)

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are AlloyRAG, an expert scientific assistant specializing in materials science, metallurgy, materials processing, microstructure, and mechanical properties.

You have access to retrieved excerpts from peer-reviewed materials-science literature.

Your goal is not merely to reproduce the retrieved excerpts. Your goal is to produce a scientifically useful, coherent, comprehensive, and literature-grounded answer that combines:

1. Clear scientific explanation
2. Mechanistic reasoning
3. Evidence from the retrieved literature
4. Quantitative results when available
5. Appropriate source attribution
6. Explicit distinction between reported evidence, scientific interpretation, and general background knowledge

The retrieved literature is the primary evidence source for literature-specific claims.

────────────────────────────────────────
SCIENTIFIC REASONING AND ANSWER CONSTRUCTION
────────────────────────────────────────

Before writing the final answer, internally determine:

- What exactly is the user asking?
- What are the main scientific dimensions that must be addressed?
- Which retrieved sources provide direct evidence for each dimension?
- Which parts can be explained using established general scientific knowledge?
- Are there alloy-, composition-, processing-, or condition-specific differences that must be preserved?
- Are there important limitations or gaps in the retrieved evidence?

Do not expose this internal reasoning process.

Construct the answer so that it addresses the major aspects of the user's question rather than merely summarizing the retrieved chunks.

For broad scientific questions, prioritize coverage of the major mechanisms, effects, and relevant distinctions.

For example, if a question asks how a heat treatment affects both microstructure and mechanical properties, address both aspects and explain the causal relationship between them.

────────────────────────────────────────
EVIDENCE HIERARCHY
────────────────────────────────────────

Use three levels of scientific statements:

A. DIRECT LITERATURE EVIDENCE
Claims directly supported by retrieved papers, tables, figures, or experimental observations.

These should be cited using the exact paper ID provided in the retrieved context.

B. SCIENTIFIC INTERPRETATION
Reasonable scientific interpretation that connects retrieved observations to established mechanisms.

Clearly distinguish interpretation from directly reported experimental findings when necessary.

C. GENERAL SCIENTIFIC BACKGROUND
Established materials-science knowledge that is useful for explaining the answer but is not specifically demonstrated by the retrieved excerpts.

Do not present general background as though it were experimentally demonstrated by the retrieved papers.

The answer should use general scientific knowledge to improve explanation and completeness, but literature-specific quantitative or experimental claims must remain grounded in the retrieved evidence.

────────────────────────────────────────
LITERATURE CITATION
────────────────────────────────────────

When a claim is supported by retrieved literature, cite the exact paper ID provided in the context.

Use the paper ID exactly as provided.

Never:
- invent a paper ID,
- modify a paper ID,
- abbreviate a paper ID,
- renumber paper IDs,
- replace paper IDs with "Source 1", "Source 2", etc.

Integrate citations naturally into the prose.

For example:

"In the retrieved D16 study [paper_id], the T6 condition increased microhardness from ..."

or:

"The 7xxx alloy study [paper_id] reported ..."

Do not attach citations to claims that are not supported by that source.

When several papers support the same conclusion, synthesize them and cite the relevant paper IDs rather than discussing each paper separately unless their results differ materially.

────────────────────────────────────────
QUANTITATIVE AND EXPERIMENTAL EVIDENCE
────────────────────────────────────────

For numerical values, experimental observations, processing parameters, phase assignments, or measured properties:

- Use retrieved literature as the source.
- Preserve the alloy/system and experimental condition.
- Do not invent missing values.
- Do not silently generalize an alloy-specific result to all materials.
- If a numerical result is available, prefer reporting it over vague statements such as "increased significantly."

Whenever useful, connect quantitative evidence to the scientific mechanism.

For example:

"The D16 study [paper_id] reported an increase in microhardness from 86.2 ± 4.1 to 135.5 ± 5.7 HV0.05 after T6 treatment. This observation is consistent with strengthening associated with precipitation evolution."

Do not claim that T6 universally produces the same magnitude of change.

────────────────────────────────────────
SCOPE AND GENERALIZATION
────────────────────────────────────────

Be careful with broad questions involving families of materials.

Do not generalize a result from one alloy, composition, or processing condition to all alloys.

Instead use language such as:

- "In the retrieved D16 study..."
- "For the investigated 7xxx alloy..."
- "The retrieved literature suggests..."
- "More generally, precipitation-hardenable Al alloys can..."
- "This observation should not be interpreted as universal because..."

If the retrieved evidence is narrow relative to the scope of the question, explicitly acknowledge this limitation.

────────────────────────────────────────
MECHANISTIC EXPLANATION
────────────────────────────────────────

Whenever appropriate, explain causal relationships rather than simply listing observations.

Prefer:

"Solution treatment dissolves soluble phases and creates a supersaturated solid solution. Rapid quenching retains this supersaturation, while artificial aging promotes precipitation. The resulting fine precipitates impede dislocation motion and can therefore increase hardness and strength."

over a disconnected list of effects.

However, do not introduce specific phase names, precipitate sequences, temperatures, or numerical values as literature findings unless they are supported by the retrieved context or are clearly presented as general scientific background.

────────────────────────────────────────
ANSWER STRUCTURE
────────────────────────────────────────

Do not use a rigid template for every question.

Choose a structure appropriate to the user's intent.

For conceptual scientific questions, a useful structure is often:

1. Direct answer / key conclusion
2. Mechanism or scientific explanation
3. Microstructural effects
4. Mechanical-property effects
5. Evidence from the retrieved literature
6. Important limitations or alloy-specific differences

Use only the sections that improve clarity.

For narrow questions, answer concisely.

For broad questions, provide sufficient depth and organization to be genuinely useful.

Use headings and bullet points when they improve readability, but avoid excessive fragmentation.

────────────────────────────────────────
ACCURACY AND UNCERTAINTY
────────────────────────────────────────

Never fabricate:
- numerical values,
- experimental conditions,
- material compositions,
- phase fractions,
- processing parameters,
- literature findings,
- citations,
- or source metadata.

If the retrieved context is insufficient to support an important literature-specific claim, say so.

If sources conflict, report the conflict and preserve the relevant experimental conditions rather than silently choosing one result.

If evidence is limited, do not compensate by making unsupported claims sound authoritative.

────────────────────────────────────────
FINAL RESPONSE QUALITY
────────────────────────────────────────

The final answer should be:

- scientifically accurate,
- coherent and logically connected,
- comprehensive relative to the question,
- mechanistically explanatory,
- concise where possible,
- quantitatively specific where evidence permits,
- literature-grounded,
- traceable to retrieved sources,
- and explicit about uncertainty and scope.

Do not merely summarize retrieved chunks.

Synthesize the evidence into a scientifically meaningful explanation.

Always prioritize the user's actual question and scientific usefulness over rigid formatting rules."""






ALLOY_QUERY_PROMPT = """User Question:
{query}

Retrieved Context:
The following excerpts were retrieved from the AlloyRAG materials-science corpus.

They may contain:
- experimental results,
- numerical data,
- tables,
- processing conditions,
- microstructural observations,
- mechanical properties,
- phase information,
- or scientific discussion.

{context}

Answer the user's question as a scientific expert.

Important requirements:
- Answer the user's question directly. 
- If the retrieved context contains partial information (e.g., UTS is found but YS is missing), provide the available data and explicitly state what is missing. Do not refuse to answer entirely if partial evidence exists.
- If the user asks for a recommendation, synthesize a recommendation based on the properties and processing routes present in the context, even if the exact grade isn't explicitly named. State the constraints that the context does not validate.
- Synthesize the retrieved evidence rather than merely listing excerpts.
- Explain relevant cause-and-effect mechanisms.
- Cite literature-supported claims using the exact paper IDs provided in the retrieved context.
- Never invent, modify, or abbreviate paper IDs.
- For numerical values, preserve the relevant alloy/system and processing conditions.
- Do not generalize an alloy-specific observation to an entire alloy family without qualification.
- Do not fabricate information, numerical values, experimental conditions, or citations.

The answer should feel like a well-written scientific explanation supported by literature evidence, not like a raw retrieval summary."""


# ── LLM Call Helper ────────────────────────────────────────────────────────────
def call_llm(system_prompt: str, user_message: str) -> str:
    """Call LLM via Gemini OpenAI-compatible endpoint."""
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"ERROR: LLM call failed — {e}"


# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk_id:    str
    text:        str
    paper_id:    str
    alloy_system: str
    chunk_type:  str        # "text" or "table"
    section:     str
    score:       float
    rank_source: str        # "dense", "bm25", or "hybrid"


@dataclass
class RAGResponse:
    query:             str
    answer:            str
    retrieved_chunks:  list[RetrievedChunk]
    n_dense_retrieved: int
    n_bm25_retrieved:  int
    n_final_chunks:    int
    alloy_filter:      Optional[str]


# ── Citation post-processor ────────────────────────────────────────────────────
def _fix_citations(answer: str, chunks: list["RetrievedChunk"]) -> str:
    """
    Deterministic post-processing: remap any degraded citation the LLM
    emits ([1], [Source 1], [SOURCE 1], [1, 2]) back to the actual paper IDs
    based on the order of retrieved chunks.

    The mapping assumes the LLM's index N refers to the Nth chunk in `chunks`,
    which matches the order used by _build_context().
    """
    if not chunks:
        return answer

    idx_to_paper = {i + 1: c.paper_id for i, c in enumerate(chunks)}

    # 1) [SOURCE N] / [Source N] / [source N]
    def repl_source(match: re.Match) -> str:
        idx = int(match.group(2))
        return f"[{idx_to_paper.get(idx, f'SOURCE {idx}')}]"

    answer = re.sub(
        r"\[(SOURCE|Source|source)\s+(\d+)\]",
        repl_source,
        answer,
    )

    # 2) [SOURCE 1, SOURCE 2] / [Source 1, Source 2] (multiple in one bracket)
    def repl_multi_source(match: re.Match) -> str:
        nums = re.findall(r"\d+", match.group(0))
        papers = [idx_to_paper.get(int(n), f"SOURCE {n}") for n in nums]
        return "[" + ", ".join(papers) + "]"

    answer = re.sub(
        r"\[(?:\s*(?:SOURCE|Source|source)\s+\d+\s*,?)+",
        repl_multi_source,
        answer,
    )

    # 3) Bare numeric citations: [1], [1, 2], [1,2,3]
    def repl_numbers(match: re.Match) -> str:
        nums = re.findall(r"\d+", match.group(1))
        papers = [idx_to_paper.get(int(n), f"SOURCE {n}") for n in nums]
        return "[" + ", ".join(papers) + "]"

    answer = re.sub(
        r"\[(\s*\d+\s*(?:,\s*\d+\s*)*)\]",
        repl_numbers,
        answer,
    )

    return answer


# ── Pipeline Class ─────────────────────────────────────────────────────────────
class AlloyRAGPipeline:

    def __init__(
        self,
        top_k:   int = 7,
        dense_k: int = 15,
        bm25_k:  int = 15,
    ):
        self.top_k   = top_k
        self.dense_k = dense_k
        self.bm25_k  = bm25_k

        # Embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")

        # ChromaDB
        logger.info("Connecting to ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection("alloyrag")
        logger.info(f"ChromaDB connected — {self.collection.count()} entries")

        # BM25
        logger.info("Loading BM25 index...")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25:      BM25Okapi  = bm25_data["bm25"]
        self.bm25_ids:  list[str]  = bm25_data["ids"]
        self.bm25_texts: list[str] = bm25_data["texts"]
        logger.info("BM25 index loaded")

        # ID maps
        logger.info("Loading ID maps...")
        with open(ID_MAP_PATH, "rb") as f:
            mapping = pickle.load(f)
        self.id_to_text: dict[str, str]  = mapping["id_to_text"]
        self.id_to_meta: dict[str, dict] = mapping["id_to_meta"]
        logger.info("ID maps loaded")

        logger.info("✅ AlloyRAG pipeline initialized")

    def _dense_retrieve(
        self,
        query: str,
        k: int,
        alloy_filter: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Dense vector retrieval from ChromaDB."""
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Only filter by alloy if specified AND use $in to also catch Unknown
        if alloy_filter:
            where_filter = {
                "$or": [
                    {"alloy_system": {"$eq": alloy_filter}},
                    {"alloy_system": {"$eq": "Unknown"}},
                ]
            }
        else:
            where_filter = None  # no filter — search everything

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": k,
            "include": ["distances", "metadatas", "documents"],
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        retrieved = []
        for chunk_id, distance in zip(
            results["ids"][0],
            results["distances"][0]
        ):
            similarity = 1.0 - distance
            retrieved.append((chunk_id, similarity))

        return retrieved

    # def _bm25_retrieve(
    #     self,
    #     query: str,
    #     k: int,
    # ) -> list[tuple[str, float]]:
    #     """BM25 sparse retrieval."""
    #     tokenized_query = query.lower().split()
    #     scores          = self.bm25.get_scores(tokenized_query)

    #     top_k_indices = sorted(
    #         range(len(scores)),
    #         key=lambda i: scores[i],
    #         reverse=True
    #     )[:k]

    #     return [
    #         (self.bm25_ids[i], float(scores[i]))
    #         for i in top_k_indices
    #         if scores[i] > 0
    #     ]
    
    def _bm25_retrieve(
        self,
        query: str,
        k: int,
        alloy_filter: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """BM25 sparse retrieval with optional alloy filtering."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        retrieved = []

        for i in ranked_indices:
            score = float(scores[i])

            if score <= 0:
                continue

            chunk_id = self.bm25_ids[i]

            if alloy_filter:
                meta = self.id_to_meta.get(chunk_id, {})
                chunk_alloy = meta.get("alloy_system", "Unknown")

                # Same policy as dense retrieval:
                # allow exact alloy + Unknown.
                if chunk_alloy not in {alloy_filter, "Unknown"}:
                    continue

            retrieved.append((chunk_id, score))

            if len(retrieved) >= k:
                break

        return retrieved

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[str, float]],
        bm25_results:  list[tuple[str, float]],
        k_constant:    int = 60,
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion: combines dense + sparse rankings.
        RRF_score(d) = sum(1 / (k + rank(d)))
        """
        rrf_scores: dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(dense_results):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0) + 1 / (k_constant + rank + 1)
            )

        for rank, (chunk_id, _) in enumerate(bm25_results):
            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0) + 1 / (k_constant + rank + 1)
            )

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query: str,
        alloy_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline.
        Returns top_k chunks after RRF reranking.
        """
        dense_results = self._dense_retrieve(query, self.dense_k, alloy_filter)
        bm25_results  = self._bm25_retrieve(query, self.bm25_k, alloy_filter=alloy_filter)

        fused      = self._reciprocal_rank_fusion(dense_results, bm25_results)
        top_chunks = fused[:self.top_k]

        retrieved_chunks = []
        # for chunk_id, rrf_score in top_chunks:
        #     text = self.id_to_text.get(chunk_id, "")
        #     meta = self.id_to_meta.get(chunk_id, {})

        #     if not text:
        #         continue

        #     retrieved_chunks.append(
        #         RetrievedChunk(
        #             chunk_id=chunk_id,
        #             text=text,
        #             paper_id=meta.get("paper_id", ""),
        #             alloy_system=meta.get("alloy_system", "Unknown"),
        #             chunk_type=meta.get("chunk_type", "text"),
        #             section=meta.get("section", "unknown"),
        #             score=rrf_score,
        #             rank_source="hybrid",
        #         )
        #     )

        # return retrieved_chunks
        
        for chunk_id, rrf_score in fused:
            text = self.id_to_text.get(chunk_id, "")
            meta = self.id_to_meta.get(chunk_id, {})

            if not text:
                continue

            if chunk_type_filter is not None:
                if meta.get("chunk_type", "text") != chunk_type_filter:
                    continue

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    paper_id=meta.get("paper_id", ""),
                    alloy_system=meta.get("alloy_system", "Unknown"),
                    chunk_type=meta.get("chunk_type", "text"),
                    section=meta.get("section", "unknown"),
                    score=rrf_score,
                    rank_source="hybrid",
                )
            )

            if len(retrieved_chunks) >= self.top_k:
                break

        return retrieved_chunks

    # def _build_context(
    #     self,
    #     chunks: list[RetrievedChunk],
    # ) -> tuple[str, int, int]:
    #     """Format retrieved chunks into LLM context string."""
    #     context_parts = []
    #     n_text = n_table = 0

    #     for i, chunk in enumerate(chunks):
    #         if chunk.chunk_type == "table":
    #             context_parts.append(
    #                 f"[SOURCE {i+1} | TABLE | Paper: {chunk.paper_id} | "
    #                 f"Alloy: {chunk.alloy_system}]\n{chunk.text}\n"
    #             )
    #             n_table += 1
    #         else:
    #             context_parts.append(
    #                 f"[SOURCE {i+1} | TEXT | Paper: {chunk.paper_id} | "
    #                 f"Section: {chunk.section} | Alloy: {chunk.alloy_system}]\n"
    #                 f"{chunk.text}\n"
    #             )
    #             n_text += 1

    #     return "\n---\n".join(context_parts), n_text, n_table
    
    def _build_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, int, int]:
        """
        Format retrieved chunks into LLM context string.

        Header format uses [PAPER: <id>] as the FIRST and ONLY identifier so the
        LLM cannot collapse citations into [1], [Source 1], etc.
        """
        context_parts = []
        n_text = n_table = 0

        for chunk in chunks:
            if chunk.chunk_type == "table":
                context_parts.append(
                    f"[PAPER: {chunk.paper_id} | TABLE | "
                    f"Alloy: {chunk.alloy_system}]\n{chunk.text}\n"
                )
                n_table += 1
            else:
                context_parts.append(
                    f"[PAPER: {chunk.paper_id} | TEXT | "
                    f"Section: {chunk.section} | Alloy: {chunk.alloy_system}]\n"
                    f"{chunk.text}\n"
                )
                n_text += 1

        return "\n---\n".join(context_parts), n_text, n_table

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Generate LLM response given query and retrieved context."""
        context, n_text, n_table = self._build_context(chunks)

        user_message = ALLOY_QUERY_PROMPT.format(
            query=query,
            n_chunks=n_text,
            n_tables=n_table,
            context=context,
        )

        raw_answer = call_llm(SYSTEM_PROMPT, user_message)
        return _fix_citations(raw_answer, chunks)

    def query(
        self,
        user_query:   str,
        alloy_filter: Optional[str] = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline: retrieve → generate.

        Args:
            user_query:   Natural language alloy design question
            alloy_filter: Optional filter ("Al", "Ti", "Steel", "Mg")
        """
        logger.info(f"Query: {user_query[:80]}...")

        chunks = self.retrieve(user_query, alloy_filter)
        answer = self.generate(user_query, chunks)

        n_tables = sum(1 for c in chunks if c.chunk_type == "table")

        return RAGResponse(
            query=user_query,
            answer=answer,
            retrieved_chunks=chunks,
            n_dense_retrieved=self.dense_k,
            n_bm25_retrieved=self.bm25_k,
            n_final_chunks=len(chunks),
            alloy_filter=alloy_filter,
        )


# ── Bare LLM Baseline ──────────────────────────────────────────────────────────
def bare_llm_answer(query: str) -> str:
    """
    Direct LLM answer WITHOUT retrieval.
    Used as hallucination baseline in evaluate.py.
    """
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a materials science expert. "
                        "Answer the following alloy design question based on "
                        "your training knowledge only. "
                        "Do not mention retrieval or databases."
                    ),
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"bare_llm_answer failed: {e}")
        return f"ERROR: {e}"


# ── Smoke Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initializing AlloyRAG pipeline...", flush=True)

    pipeline = AlloyRAGPipeline(top_k=5)

    test_query = (
        "What is the effect of T6 vs T73 temper on the mechanical properties and corrosion resistance of 7075 aluminum alloy?"
    )

    print(f"\nQuery: {test_query}\n", flush=True)
    print("Retrieving and generating...", flush=True)

    result = pipeline.query(test_query)

    print("\n" + "="*60, flush=True)
    print("AlloyRAG Response:", flush=True)
    print("="*60, flush=True)
    print(result.answer, flush=True)
    print("="*60, flush=True)
    print(
        f"Retrieved {result.n_final_chunks} chunks "
        f"({sum(1 for c in result.retrieved_chunks if c.chunk_type == 'table')} tables)",
        flush=True
    )

    print("\nTesting bare LLM baseline...", flush=True)
    bare = bare_llm_answer(test_query)
    print("\nBare LLM Response:", flush=True)
    print("="*60, flush=True)
    print(bare, flush=True)
    print("="*60, flush=True)