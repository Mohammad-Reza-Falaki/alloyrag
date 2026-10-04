"""
Module 1
Multi-source paper collection — no API key required
Sources: Crossref + arXiv + Unpaywall (PDF finder)
"""

import os
import re
import time
import json
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
# Unpaywall needs your email — no registration, just an identifier
YOUR_EMAIL = "email@gmail.com"  # ← change this to your real email

OUTPUT_DIR = Path("data/raw_metadata")
PDF_DIR = Path("data/pdfs")

TARGET_COUNTS = {"Al": 100, "Ti": 65, "Steel": 80, "Mg": 55}

# Search queries per alloy system
SEARCH_QUERIES = {
    "Al": [
        "aluminum alloy microstructure mechanical properties tensile",
        "7075 aluminum alloy aging heat treatment strength",
        "2024 Al alloy fatigue corrosion microstructure",
        "aluminum alloy additive manufacturing SLM properties",
        "Al-Zn-Mg alloy precipitation hardening",
        "6061 aluminum alloy processing microstructure",
        "aluminum alloy severe plastic deformation ECAP",
        "Al alloy welding friction stir properties",
        "cast aluminum alloy solidification microstructure",
        "aluminum alloy nanocomposite mechanical properties",
        "5xxx aluminum alloy corrosion marine",
        "2xxx aluminum alloy aerospace fatigue",
    ],
    "Ti": [
        "Ti-6Al-4V mechanical properties microstructure",
        "titanium alloy additive manufacturing fatigue",
        "beta titanium alloy heat treatment tensile",
        "titanium alloy biomedical implant corrosion",
        "near alpha titanium alloy creep properties",
        "titanium alloy forging microstructure texture",
        "titanium aluminide intermetallic high temperature",
        "titanium alloy surface treatment oxidation",
        "Ti alloy selective laser melting porosity",
    ],
    "Steel": [
        "dual phase steel microstructure tensile deformation",
        "high strength steel martensite bainite properties",
        "stainless steel corrosion mechanical properties",
        "TRIP steel retained austenite transformation",
        "HSLA steel thermomechanical processing toughness",
        "advanced high strength steel automotive forming",
        "martensitic steel tempering microstructure hardness",
        "tool steel heat treatment wear properties",
        "bearing steel fatigue rolling contact",
        "pipeline steel sour service hydrogen",
    ],
    "Mg": [
        "magnesium alloy deformation texture mechanical",
        "AZ31 AZ91 magnesium alloy microstructure processing",
        "Mg alloy corrosion biodegradable implant",
        "magnesium alloy rare earth ductility",
        "Mg alloy extrusion twinning properties",
        "magnesium alloy die casting automotive",
        "WE43 ZK60 magnesium alloy properties",
    ],
}


@dataclass
class PaperRecord:
    paper_id: str
    title: str
    authors: list
    year: Optional[int]
    doi: Optional[str]
    abstract: Optional[str]
    pdf_url: Optional[str]
    journal: Optional[str]
    alloy_system: str
    query_matched: str
    source: str  # "crossref" or "arxiv"


# ── Helper ─────────────────────────────────────────────────────────────────────

def safe_get(url: str, params: dict = None, timeout: int = 20) -> Optional[dict]:
    """
    Simple GET with retry and backoff.
    Returns parsed JSON or None on failure.
    """
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                logger.warning(f"429 rate limit — waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                wait = 2 ** (attempt + 3)
                logger.warning(f"403 forbidden — waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for {url[:60]}")
                return None

            return resp.json()

        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"Request error: {e}")
            return None

    return None


# ── Source 1: Crossref ─────────────────────────────────────────────────────────

def search_crossref(query: str, alloy_system: str, limit: int = 20) -> list[PaperRecord]:
    """
    Search Crossref API — completely free, no key needed.
    Focuses on journal articles with abstracts.
    Rate limit: polite pool = ~50 req/sec with email header
    """
    url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": limit,
        "filter": "type:journal-article,from-pub-date:2015",
        "select": "DOI,title,author,published,abstract,container-title,link",
        "mailto": YOUR_EMAIL,  # gets you into polite pool — faster responses
    }

    data = safe_get(url, params=params)
    if not data:
        return []

    records = []
    items = data.get("message", {}).get("items", [])

    for item in items:
        title_list = item.get("title", [])
        if not title_list:
            continue
        title = title_list[0].strip()

        abstract = item.get("abstract", "")
        if abstract:
            # Crossref wraps abstracts in JATS XML tags — clean them
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        if not abstract or len(abstract) < 80:
            continue  # skip papers without usable abstract

        doi = item.get("DOI", "")

        # Extract year
        year = None
        pub_date = item.get("published", {})
        date_parts = pub_date.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        # Extract authors
        authors = []
        for a in item.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            if family:
                authors.append(f"{given} {family}".strip())

        # Extract journal name
        journal_list = item.get("container-title", [])
        journal = journal_list[0] if journal_list else None

        # Try to find PDF link from Crossref links
        pdf_url = None
        for link in item.get("link", []):
            if link.get("content-type") == "application/pdf":
                pdf_url = link.get("URL")
                break

        paper_id = f"CR_{doi.replace('/', '_').replace('.', '_')}" if doi else f"CR_{hash(title)}"

        records.append(PaperRecord(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            abstract=abstract,
            pdf_url=pdf_url,
            journal=journal,
            alloy_system=alloy_system,
            query_matched=query,
            source="crossref",
        ))

    logger.info(f"  Crossref '{query[:45]}' → {len(records)} records")
    return records


# ── Source 2: arXiv ────────────────────────────────────────────────────────────

def search_arxiv(query: str, alloy_system: str, limit: int = 15) -> list[PaperRecord]:
    """
    Search arXiv API — completely free, no key needed.
    Materials science papers are under cond-mat.mtrl-sci
    All arXiv papers have free PDF download.
    """
    import xml.etree.ElementTree as ET

    url = "http://export.arxiv.org/api/query"
    # Search in materials science category + general query
    params = {
        "search_query": f"cat:cond-mat.mtrl-sci AND all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"arXiv returned {resp.status_code}")
            return []
    except Exception as e:
        logger.warning(f"arXiv request failed: {e}")
        return []

    records = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    try:
        root = ET.fromstring(resp.content)
        entries = root.findall("atom:entry", ns)
    except ET.ParseError as e:
        logger.warning(f"arXiv XML parse error: {e}")
        return []

    for entry in entries:
        title_el = entry.find("atom:title", ns)
        abstract_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)

        if title_el is None or abstract_el is None:
            continue
        
        title = title_el.text.strip().replace("\n", " ")
        abstract = abstract_el.text.strip().replace("\n", " ")

        if len(abstract) < 80:
            continue

        # arXiv ID from URL like http://arxiv.org/abs/2301.12345v1
        arxiv_id = ""
        if id_el is not None:
            arxiv_id = id_el.text.strip().split("/abs/")[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)  # remove version

        # PDF URL is always available for arXiv
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None

        # Published date
        published_el = entry.find("atom:published", ns)
        year = None
        if published_el is not None:
            try:
                year = int(published_el.text[:4])
            except (ValueError, TypeError):
                pass

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None:
                authors.append(name_el.text.strip())

        paper_id = f"AX_{arxiv_id.replace('.', '_').replace('-', '_')}"

        records.append(PaperRecord(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            doi=f"arxiv:{arxiv_id}",
            abstract=abstract,
            pdf_url=pdf_url,
            journal="arXiv preprint",
            alloy_system=alloy_system,
            query_matched=query,
            source="arxiv",
        ))

    logger.info(f"  arXiv '{query[:45]}' → {len(records)} records")
    time.sleep(3)  # arXiv asks for 3s between requests
    return records


# ── Source 3: Unpaywall PDF Finder ─────────────────────────────────────────────

def find_pdf_unpaywall(doi: str) -> Optional[str]:
    """
    Given a DOI, find a free legal PDF using Unpaywall.
    No API key needed — just your email as identifier.
    Returns PDF URL or None.
    """
    if not doi or not YOUR_EMAIL or YOUR_EMAIL == "your_email@example.com":
        return None

    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": YOUR_EMAIL}

    data = safe_get(url, params=params)
    if not data:
        return None

    # Try best OA location first
    best_location = data.get("best_oa_location")
    if best_location:
        pdf_url = best_location.get("url_for_pdf")
        if pdf_url:
            return pdf_url

    # Try all OA locations
    for location in data.get("oa_locations", []):
        pdf_url = location.get("url_for_pdf")
        if pdf_url:
            return pdf_url

    return None


# ── Main Collection ────────────────────────────────────────────────────────────

def collect_all_papers() -> dict:
    """
    Collect papers from Crossref + arXiv.
    Saves progress after every query.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    existing_path = OUTPUT_DIR / "papers_metadata.json"
    seen_ids: set = set()
    seen_titles: set = set()  # extra dedup by title
    collected: dict = {k: [] for k in TARGET_COUNTS}

    # Resume support
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)
        for record in existing:
            alloy = record.get("alloy_system", "")
            if alloy in collected:
                collected[alloy].append(record)
                seen_ids.add(record["paper_id"])
                seen_titles.add(record.get("title", "").lower()[:60])
        logger.info(f"Resumed from {len(seen_ids)} existing papers")

    def save_progress():
        all_records = [
            r if isinstance(r, dict) else asdict(r)
            for records in collected.values()
            for r in records
        ]
        with open(existing_path, "w") as f:
            json.dump(all_records, f, indent=2)
        return len(all_records)

    # Collection loop
    for alloy_system, queries in SEARCH_QUERIES.items():
        target = TARGET_COUNTS[alloy_system]
        
        if len(collected[alloy_system]) >= target:
            logger.info(f"{alloy_system}: already complete ({target} papers)")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Collecting {alloy_system} papers (target: {target})")
        logger.info(f"{'='*50}")

        for query in queries:
            if len(collected[alloy_system]) >= target:
                break

            # Try Crossref first
            crossref_results = search_crossref(query, alloy_system, limit=20)
            time.sleep(1.0)

            # Then arXiv
            arxiv_results = search_arxiv(query, alloy_system, limit=10)
            # arXiv sleep is inside the function

            all_results = crossref_results + arxiv_results

            added = 0
            for record in all_results:
                if len(collected[alloy_system]) >= target:
                    break

                rid = record.paper_id if hasattr(record, "paper_id") else record["paper_id"]
                rtitle = (record.title if hasattr(record, "title") else record["title"]).lower()[:60]

                if rid in seen_ids or rtitle in seen_titles:
                    continue

                collected[alloy_system].append(asdict(record) if hasattr(record, "paper_id") else record)
                seen_ids.add(rid)
                seen_titles.add(rtitle)
                added += 1

            total_saved = save_progress()
            logger.info(
                f"  [{alloy_system}] {len(collected[alloy_system])}/{target} | "
                f"+{added} this query | Total: {total_saved}"
            )

    total = save_progress()
    logger.info(f"\n✅ Collection complete: {total} papers")
    for alloy, records in collected.items():
        logger.info(f"  {alloy}: {len(records)}/{TARGET_COUNTS[alloy]}")

    return collected


# ── PDF Download ───────────────────────────────────────────────────────────────

def download_pdfs() -> dict:
    """
    Download PDFs for all collected papers.
    Strategy:
      1. Use pdf_url from collection (direct OA or arXiv)
      2. Try Unpaywall for DOIs without PDF URL
      3. Log failures for manual download
    """
    metadata_path = OUTPUT_DIR / "papers_metadata.json"
    if not metadata_path.exists():
        logger.error("No papers_metadata.json found. Run collection first.")
        return {}

    with open(metadata_path) as f:
        papers = json.load(f)

    downloaded: dict = {}
    failed: list = []

    # Load existing pdf_map to support resume
    pdf_map_path = OUTPUT_DIR / "pdf_map.json"
    if pdf_map_path.exists():
        with open(pdf_map_path) as f:
            downloaded = json.load(f)

    logger.info(f"\nDownloading PDFs for {len(papers)} papers...")

    for i, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        doi = paper.get("doi", "")
        pdf_url = paper.get("pdf_url", "")

        # Skip already downloaded
        pdf_path = PDF_DIR / f"{paper_id}.pdf"
        if paper_id in downloaded and pdf_path.exists():
            continue

        # Try Unpaywall if no direct PDF URL
        if not pdf_url and doi and not doi.startswith("arxiv:"):
            logger.info(f"  [{i+1}/{len(papers)}] Trying Unpaywall for DOI: {doi[:40]}")
            pdf_url = find_pdf_unpaywall(doi)
            time.sleep(1.0)  # Unpaywall rate limit

        if not pdf_url:
            failed.append({
                "paper_id": paper_id,
                "title": paper.get("title", "")[:80],
                "doi": doi,
                "journal": paper.get("journal", ""),
            })
            continue

        # Download PDF
        success = False
        for attempt in range(3):
            try:
                resp = requests.get(
                    pdf_url,
                    timeout=45,
                    stream=True,
                    headers={"User-Agent": "Mozilla/5.0 (academic research)"},
                )
                resp.raise_for_status()
                
                with open(pdf_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                if pdf_path.stat().st_size > 5000:  # >5KB = valid PDF
                    downloaded[paper_id] = str(pdf_path)
                    logger.info(f"  [{i+1}/{len(papers)}] ✓ {paper_id}")
                    success = True
                    break
                else:
                    pdf_path.unlink(missing_ok=True)

            except Exception as e:
                time.sleep(2 ** attempt)

        if not success:
            failed.append({
                "paper_id": paper_id,
                "title": paper.get("title", "")[:80],
                "doi": doi,
                "pdf_url_tried": pdf_url[:60] if pdf_url else "",
            })

        # Save map periodically
        if (i + 1) % 10 == 0:
            with open(pdf_map_path, "w") as f:
                json.dump(downloaded, f, indent=2)

        time.sleep(0.5)

    # Final save
    with open(pdf_map_path, "w") as f:
        json.dump(downloaded, f, indent=2)

    # Save manual download list — formatted for easy reading
    if failed:
        failed_path = OUTPUT_DIR / "manual_download_needed.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        logger.info(f"\n  {len(failed)} papers need manual download")
        logger.info(f"  See: {failed_path}")
        logger.info(f"  Get PDFs from: Google Scholar, ResearchGate, author pages")
        logger.info(f"  Save as: data/pdfs/{{paper_id}}.pdf")
        logger.info(f"  Then add to: data/raw_metadata/pdf_map.json")

    logger.info(f"\n✅ Download complete: {len(downloaded)} PDFs, {len(failed)} need manual download")
    return downloaded


if __name__ == "__main__":
    # Step 1 — change your email first
    if YOUR_EMAIL == "your_email@example.com":
        print("⚠️  Please set YOUR_EMAIL at the top of collect_papers.py first")
        print("    It is used by Crossref polite pool and Unpaywall — no registration needed")
        exit(1)

    # Step 2 — collect metadata
    collect_all_papers()

    # Step 3 — download PDFs
    download_pdfs()