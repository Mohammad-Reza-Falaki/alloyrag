"""
register_manual_pdfs.py
Run this after placing manual PDFs in data/pdfs/
It scans the folder and updates pdf_map.json automatically
Save as: src/register_manual_pdfs.py
"""

import json
from pathlib import Path

PDF_DIR = Path("data/pdfs")
PDF_MAP_PATH = Path("data/raw_metadata/pdf_map.json")

# Load existing map
if PDF_MAP_PATH.exists():
    with open(PDF_MAP_PATH) as f:
        pdf_map = json.load(f)
else:
    pdf_map = {}

# Scan pdfs folder for all pdf files
added = 0
for pdf_file in PDF_DIR.glob("*.pdf"):
    paper_id = pdf_file.stem  # filename without .pdf extension
    if paper_id not in pdf_map:
        pdf_map[paper_id] = str(pdf_file)
        print(f"  Added: {paper_id}")
        added += 1
    else:
        print(f"  Already registered: {paper_id}")

# Save updated map
with open(PDF_MAP_PATH, "w") as f:
    json.dump(pdf_map, f, indent=2)

print(f"\n✅ Done: {added} new PDFs registered")
print(f"   Total in pdf_map.json: {len(pdf_map)}")