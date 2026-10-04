"""
Module 3: Build ChromaDB vector store + BM25 index
Embeds text chunks AND property table representations
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, fast, good quality
CHROMA_DB_PATH = "data/chroma_db"
BM25_INDEX_PATH = "data/bm25_index.pkl"

# Metadata fields to store alongside each chunk in ChromaDB
CHUNK_METADATA_FIELDS = [
    "paper_id", "alloy_system", "section", "year", "journal",
    "has_properties", "chunk_type"  # "text" or "table"
]


def load_all_documents(corpus_dir: str = "data/corpus") -> list[dict]:
    """Load all processed paper JSONs from corpus directory."""
    docs = []
    failed = []

    for p in Path(corpus_dir).glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:  # ← encoding fix
                docs.append(json.load(f))
        except UnicodeDecodeError:
            # Try with errors='replace' as fallback
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    docs.append(json.load(f))
                logger.warning(f"Loaded with replacement chars: {p.name}")
            except Exception as e:
                logger.warning(f"Failed to load {p.name}: {e}")
                failed.append(p.name)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {p.name}: {e}")
            failed.append(p.name)
        except Exception as e:
            logger.warning(f"Unexpected error loading {p.name}: {e}")
            failed.append(p.name)

    if failed:
        logger.warning(f"Could not load {len(failed)} files: {failed[:5]}")

    logger.info(f"Loaded {len(docs)} documents from corpus")
    return docs


def prepare_retrieval_units(docs: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """
    Prepare all retrieval units: text chunks + table representations.
    
    Returns:
        ids: unique IDs for each unit
        metadatas: metadata dicts for ChromaDB filtering
        texts: text content to embed
    """
    ids, metadatas, texts = [], [], []

    for doc in docs:
        paper_id = doc["paper_id"]
        alloy_system = doc.get("alloy_system", "Unknown")
        year = doc.get("year") or 0
        journal = doc.get("journal") or "Unknown"
        has_properties = len(doc.get("properties_extracted", {})) > 0

        # TEXT CHUNKS
        for chunk in doc.get("text_chunks", []):
            chunk_text = chunk["text"].strip()
            if len(chunk_text) < 50:
                continue

            ids.append(chunk["chunk_id"])
            texts.append(chunk_text)
            metadatas.append({
                "paper_id": paper_id,
                "alloy_system": alloy_system,
                "section": chunk.get("section", "unknown"),
                "year": year,
                "journal": journal,
                "has_properties": has_properties,
                "chunk_type": "text",
                "title": doc.get("title", "")[:200],
            })

        # TABLE REPRESENTATIONS (indexed as special retrieval units)
        for table in doc.get("property_tables", []):
            text_repr = table.get("text_representation", "").strip()
            if len(text_repr) < 20:
                continue

            # Prefix with context so embeddings know this is a property table
            alloy_desigs = ", ".join(doc.get("alloy_designations", [])[:5])
            enriched = (
                f"Materials property data table. "
                f"Alloy system: {alloy_system}. "
                f"Alloy designations: {alloy_desigs if alloy_desigs else 'not specified'}. "
                f"Source paper: {doc.get('title','')[:100]}. "
                f"Year: {year}. "
                f"Mechanical and physical properties: {text_repr}"
            )

            ids.append(table["table_id"])
            texts.append(enriched)
            metadatas.append({
                "paper_id": paper_id,
                "alloy_system": alloy_system,
                "section": "table",
                "year": year,
                "journal": journal,
                "has_properties": True,
                "chunk_type": "table",
                "title": doc.get("title", "")[:200],
            })

    logger.info(f"Prepared {len(ids)} retrieval units ({sum(1 for m in metadatas if m['chunk_type']=='table')} tables)")
    return ids, metadatas, texts


def build_chroma_index(
    ids: list[str],
    metadatas: list[dict],
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 128,
) -> chromadb.Collection:
    """Build and persist ChromaDB collection with embeddings."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("alloyrag")
    except Exception:
        pass

    collection = client.create_collection(
        name="alloyrag",
        metadata={"hnsw:space": "cosine"},
    )

    logger.info(f"Embedding {len(texts)} units with {EMBEDDING_MODEL}...")

    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]
        batch_metas = metadatas[batch_start:batch_end]

        embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)

        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metas,
        )

        if (batch_start // batch_size) % 5 == 0:
            logger.info(f"  Indexed {batch_end}/{len(texts)} units")

    logger.info(f"✅ ChromaDB index built: {collection.count()} entries")
    return collection


def build_bm25_index(texts: list[str], ids: list[str]) -> dict:
    """Build BM25 sparse index for keyword-based retrieval."""
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    index_data = {"bm25": bm25, "ids": ids, "texts": texts}
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    logger.info(f"✅ BM25 index built and saved to {BM25_INDEX_PATH}")
    return index_data


def build_full_index(corpus_dir: str = "data/corpus") -> None:
    """End-to-end index building pipeline."""
    docs = load_all_documents(corpus_dir)
    ids, metadatas, texts = prepare_retrieval_units(docs)

    model = SentenceTransformer(EMBEDDING_MODEL)

    build_chroma_index(ids, metadatas, texts, model)
    build_bm25_index(texts, ids)

    # Save id→text mapping for BM25 result lookup
    id_to_text = dict(zip(ids, texts))
    id_to_meta = dict(zip(ids, metadatas))
    with open("data/id_to_text.pkl", "wb") as f:
        pickle.dump({"id_to_text": id_to_text, "id_to_meta": id_to_meta}, f)

    logger.info("✅ All indexes built successfully")


if __name__ == "__main__":
    import sys
    import logging

    # Force all output to show immediately in CMD
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    logger = logging.getLogger(__name__)

    print("STEP 1: Script started", flush=True)
    sys.stdout.flush()

    # ── Test 1: Can we find corpus files? ─────────────────────────────────────
    from pathlib import Path
    corpus_path = Path("data/corpus")
    print(f"STEP 2: Looking for corpus at: {corpus_path.absolute()}", flush=True)
    print(f"STEP 3: Corpus exists: {corpus_path.exists()}", flush=True)

    if not corpus_path.exists():
        print("ERROR: data/corpus folder not found", flush=True)
        print("Make sure you run from AlloyRAG root:", flush=True)
        print("  cd AlloyRAG", flush=True)
        print("  python src/build_index.py", flush=True)
        sys.exit(1)

    files = list(corpus_path.glob("*.json"))
    print(f"STEP 4: Found {len(files)} JSON files in corpus", flush=True)

    if len(files) == 0:
        print("ERROR: No JSON files found in data/corpus/", flush=True)
        print("Run extract_papers.py first", flush=True)
        sys.exit(1)

    # ── Test 2: Can we load one document? ─────────────────────────────────────
    import json
    print(f"STEP 5: Testing load of first corpus file...", flush=True)
    try:
        with open(files[0]) as f:
            sample = json.load(f)
        print(f"STEP 6: Sample file loaded OK: {sample.get('paper_id', 'unknown')}", flush=True)
        print(f"        Chunks in sample: {len(sample.get('text_chunks', []))}", flush=True)
        print(f"        Tables in sample: {len(sample.get('property_tables', []))}", flush=True)
    except Exception as e:
        print(f"ERROR loading corpus file: {e}", flush=True)
        sys.exit(1)

    # ── Test 3: Load all documents ─────────────────────────────────────────────
    print(f"STEP 7: Loading all {len(files)} corpus files...", flush=True)
    try:
        docs = load_all_documents("data/corpus")
        print(f"STEP 8: All documents loaded: {len(docs)}", flush=True)
    except Exception as e:
        print(f"ERROR in load_all_documents: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Test 4: Prepare retrieval units ───────────────────────────────────────
    print(f"STEP 9: Preparing retrieval units...", flush=True)
    try:
        ids, metadatas, texts = prepare_retrieval_units(docs)
        print(f"STEP 10: Retrieval units prepared: {len(ids)}", flush=True)
        print(f"         Text units:  {sum(1 for m in metadatas if m['chunk_type']=='text')}", flush=True)
        print(f"         Table units: {sum(1 for m in metadatas if m['chunk_type']=='table')}", flush=True)
    except Exception as e:
        print(f"ERROR in prepare_retrieval_units: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Test 5: Load embedding model ───────────────────────────────────────────
    print(f"STEP 11: Loading SentenceTransformer model...", flush=True)
    print(f"         (may take 1-3 minutes first time — downloading ~90MB)", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"STEP 12: Model loaded OK: {EMBEDDING_MODEL}", flush=True)
    except Exception as e:
        print(f"ERROR loading SentenceTransformer: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Test 6: Test embed one batch ───────────────────────────────────────────
    print(f"STEP 13: Testing embedding on 5 sample texts...", flush=True)
    try:
        sample_texts = texts[:5]
        sample_embeddings = model.encode(
            sample_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"STEP 14: Embedding test OK — shape: {sample_embeddings.shape}", flush=True)
    except Exception as e:
        print(f"ERROR during embedding test: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Test 7: Build ChromaDB ─────────────────────────────────────────────────
    print(f"STEP 15: Building ChromaDB index...", flush=True)
    try:
        collection = build_chroma_index(ids, metadatas, texts, model)
        print(f"STEP 16: ChromaDB built OK — {collection.count()} entries", flush=True)
    except Exception as e:
        print(f"ERROR building ChromaDB: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Test 8: Build BM25 ────────────────────────────────────────────────────
    print(f"STEP 17: Building BM25 index...", flush=True)
    try:
        bm25_data = build_bm25_index(texts, ids)
        print(f"STEP 18: BM25 built OK", flush=True)
    except Exception as e:
        print(f"ERROR building BM25: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("", flush=True)
    print("=" * 50, flush=True)
    print("✅ build_index.py completed successfully", flush=True)
    print("=" * 50, flush=True)