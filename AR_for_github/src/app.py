"""
Module 7: Gradio web app demo
Live demo for paper — retrieval + generation + source attribution.

Safety / read-only guarantee
────────────────────────────
Reads (only):
  • .env                                       (via load_dotenv)
  • data/chroma_db/                            (via ChromaDB client)
  • data/bm25_index.pkl                        (via pickle.load)
  • data/id_to_text.pkl                        (via pickle.load)
  • HuggingFace model cache for all-MiniLM-L6-v2

Writes:
  • Nothing into corpus, metadata, evaluation, or index JSONs/pickles.
  • Chroma's PersistentClient may touch its own internal WAL/metadata
    files inside data/chroma_db/ — that is Chroma's own bookkeeping,
    not a modification of any project data file.

Network:
  • Gemini API via OpenAI-compatible endpoint (uses your API key).
  • HuggingFace download on first-ever run if the model is not cached.
"""

import argparse
import logging
import sys
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from rag_pipeline import AlloyRAGPipeline, bare_llm_answer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


EXAMPLE_QUERIES = [
    "I need an Al alloy with UTS > 400 MPa and good corrosion resistance for automotive use",
    "What processing route gives Ti-6Al-4V the best fatigue resistance?",
    "Which steel microstructure offers the best trade-off between strength and ductility?",
    "Recommend a Mg alloy for biodegradable biomedical implants",
    "What aging treatment maximizes the yield strength of 7075 aluminum alloy?",
]


# ── Lazy pipeline singleton ───────────────────────────────────────────────────
_pipeline: Optional[AlloyRAGPipeline] = None


def _get_pipeline() -> AlloyRAGPipeline:
    """Load the AlloyRAG pipeline on first use — not at module import."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading AlloyRAG pipeline (first call)...")
        _pipeline = AlloyRAGPipeline(top_k=5)
        logger.info("AlloyRAG pipeline ready.")
    return _pipeline


# ── Query handler ─────────────────────────────────────────────────────────────
def query_alloyrag(
    user_query: str,
    alloy_filter: str,
    show_baseline: bool,
) -> tuple[str, str, str]:
    """
    Gradio callback.
    Returns (answer_markdown, sources_markdown, baseline_markdown).
    Never raises — every error is surfaced as Markdown so the UI stays alive.
    """
    if not user_query or not user_query.strip():
        return "Please enter a query.", "", ""

    filter_val = alloy_filter if alloy_filter != "All" else None

    # ── RAG pipeline ───────────────────────────────────────────────────────
    try:
        result = _get_pipeline().query(user_query, alloy_filter=filter_val)
    except Exception as e:
        logger.exception("AlloyRAG query failed")
        return f"⚠️ Pipeline error: `{type(e).__name__}: {e}`", "", ""

    # ── Format retrieved sources ───────────────────────────────────────────
    if result.retrieved_chunks:
        sources_text = "### Retrieved Sources\n\n"
        for i, chunk in enumerate(result.retrieved_chunks):
            icon    = "📊" if chunk.chunk_type == "table" else "📄"
            preview = chunk.text[:600].replace("\n", " ")
            sources_text += (
                f"**Source {i+1}** {icon} | "
                f"Paper: `{chunk.paper_id}` | "
                f"Alloy: {chunk.alloy_system} | "
                f"Section: {chunk.section}\n\n"
                f"> {preview}...\n\n---\n"
            )
    else:
        sources_text = "_No sources retrieved._"

    # ── Optional bare-LLM baseline ─────────────────────────────────────────
    baseline_text = ""
    if show_baseline:
        try:
            baseline = bare_llm_answer(user_query)
        except Exception as e:
            baseline = f"⚠️ Bare LLM failed: `{type(e).__name__}: {e}`"
        baseline_text = f"### Bare LLM (No Retrieval)\n\n{baseline}"

    return result.answer, sources_text, baseline_text


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="AlloyRAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 AlloyRAG
    ### LLM-Powered Retrieval-Augmented Generation for Alloy Design
    *Recommendations grounded in peer-reviewed materials science literature*
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Alloy Design Query",
                placeholder="e.g., I need an Al alloy with UTS > 400 MPa for automotive use...",
                lines=3,
            )
        with gr.Column(scale=1):
            alloy_filter = gr.Dropdown(
                choices=["All", "Al", "Ti", "Steel", "Mg"],
                value="All",
                label="Prefer Alloy System (may still return 'Unknown' chunks)",
            )
            show_baseline = gr.Checkbox(
                label="Show Bare LLM Baseline (for comparison)",
                value=False,
            )

    submit_btn = gr.Button("🔍 Query AlloyRAG", variant="primary")

    gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_input)

    with gr.Tabs():
        with gr.TabItem("📋 Recommendation"):
            answer_output = gr.Markdown()
        with gr.TabItem("📚 Retrieved Sources"):
            sources_output = gr.Markdown()
        with gr.TabItem("🤖 Baseline Comparison"):
            baseline_output = gr.Markdown()

    submit_btn.click(
        fn=query_alloyrag,
        inputs=[query_input, alloy_filter, show_baseline],
        outputs=[answer_output, sources_output, baseline_output],
    )


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AlloyRAG Gradio demo")
    ap.add_argument(
        "--share", action="store_true",
        help="Expose a public Gradio URL. WARNING: uses your LLM API quota.",
    )
    ap.add_argument(
        "--port", type=int, default=7860,
        help="Local port (default: 7860).",
    )
    args = ap.parse_args()

    demo.launch(share=args.share, server_port=args.port)