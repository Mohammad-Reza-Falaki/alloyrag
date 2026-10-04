"""
rebuild_idmap.py
Rebuilds id_to_text.pkl and id_to_meta.pkl from existing corpus
without re-embedding anything
Run: python src/rebuild_idmap.py
"""

import json
import pickle
import sys
from pathlib import Path

print("Loading corpus files...", flush=True)

corpus_dir = Path("data/corpus")
if not corpus_dir.exists():
    print("ERROR: data/corpus not found")
    sys.exit(1)

files = list(corpus_dir.glob("*.json"))
print(f"Found {len(files)} corpus files", flush=True)

id_to_text = {}
id_to_meta = {}

for i, path in enumerate(files):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    paper_id     = doc["paper_id"]
    alloy_system = doc.get("alloy_system", "Unknown")
    year         = doc.get("year") or 0
    journal      = doc.get("journal") or "Unknown"
    title        = doc.get("title", "")[:200]

    # Text chunks
    for chunk in doc.get("text_chunks", []):
        text = chunk["text"].strip()
        if len(text) < 50:
            continue
        cid = chunk["chunk_id"]
        id_to_text[cid] = text
        id_to_meta[cid] = {
            "paper_id":     paper_id,
            "alloy_system": alloy_system,
            "section":      chunk.get("section", "unknown"),
            "year":         year,
            "journal":      journal,
            "chunk_type":   "text",
            "title":        title,
        }

    # Table units
    for table in doc.get("property_tables", []):
        text_repr = table.get("text_representation", "").strip()
        if len(text_repr) < 20:
            continue
        # enriched = (
        #     f"Property table from: {doc.get('title','')[:100]}. "
        #     f"Alloy system: {alloy_system}. "
        #     f"Data: {text_repr}"
        # )
        alloy_desigs_str = ", ".join(doc.get("alloy_designations", [])[:5])
        enriched = (
            f"Materials property data table. "
            f"Alloy system: {alloy_system}. "
            f"Alloy designations: {alloy_desigs_str if alloy_desigs_str else 'not specified'}. "
            f"Source paper: {doc.get('title','')[:100]}. "
            f"Year: {year}. "
            f"Mechanical and physical properties: {text_repr}"
        )
        tid = table["table_id"]
        id_to_text[tid] = enriched
        id_to_meta[tid] = {
            "paper_id":     paper_id,
            "alloy_system": alloy_system,
            "section":      "table",
            "year":         year,
            "journal":      journal,
            "chunk_type":   "table",
            "title":        title,
        }

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(files)} files...", flush=True)

# Save
output_path = Path("data/id_to_text.pkl")
with open(output_path, "wb") as f:
    pickle.dump({"id_to_text": id_to_text, "id_to_meta": id_to_meta}, f)

print(f"\n✅ Done", flush=True)
print(f"   Total text entries:  {sum(1 for v in id_to_meta.values() if v['chunk_type']=='text')}", flush=True)
print(f"   Total table entries: {sum(1 for v in id_to_meta.values() if v['chunk_type']=='table')}", flush=True)
print(f"   Total entries:       {len(id_to_text)}", flush=True)
print(f"   Saved to:            {output_path.absolute()}", flush=True)