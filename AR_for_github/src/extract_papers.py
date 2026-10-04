"""
Module 2: Robust PDF extraction pipeline - Final Version
Improvements over previous version:
1. Text-strategy fallback for borderless tables
2. Multi-page table merging for split tables
3. Enhanced unit-based type detection (Greek-header tables)
4. Fuzzy caption matching by content proximity
5. Confidence threshold with numerical data verification
"""

import json
import re
import logging
import sys
from pathlib import Path
from typing import Optional

import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfparser import PDFSyntaxError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# GREEK SYMBOL NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

GREEK_MAP = {
    'Σ': 'sigma',   'Ε': 'epsilon', 'Δ': 'delta',   'Ω': 'omega',
    'Α': 'alpha',   'Β': 'beta',    'Γ': 'gamma',   'Λ': 'lambda',
    'σ': 'sigma',   'ε': 'epsilon', 'δ': 'delta',   'ω': 'omega',
    'α': 'alpha',   'β': 'beta',    'γ': 'gamma',   'λ': 'lambda',
    'μ': 'mu',      'ρ': 'rho',     'τ': 'tau',     'φ': 'phi',
    '\u03c3': 'sigma',   '\u03b5': 'epsilon',
    '\u03b1': 'alpha',   '\u03bc': 'mu',
    '\u03c1': 'rho',     '\u03b4': 'delta',
    '\u03b8': 'theta',   '\u03bd': 'nu',
    '\u03c7': 'chi',     '\u03c8': 'psi',
}

SUBSCRIPT_MAP = {
    '\u2080': '0', '\u2081': '1', '\u2082': '2',
    '\u2083': '3', '\u2084': '4', '\u2085': '5',
    '\u00b2': '2', '\u00b3': '3',
    '\u207b': '-',  # superscript minus
    '\u00b9': '1',  # superscript 1
}


def normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching.
    Converts Greek letters, subscripts, and special chars to ASCII.
    Also handles common PDF encoding issues like ligatures.
    """
    if not text:
        return ""
    result = text
    for char, replacement in GREEK_MAP.items():
        result = result.replace(char, replacement)
    for char, replacement in SUBSCRIPT_MAP.items():
        result = result.replace(char, replacement)
    # Fix common ligatures
    result = result.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    result = result.replace('ﬀ', 'ff').replace('ﬃ', 'ffi')
    # Normalize spaces
    result = re.sub(r'\s+', ' ', result)
    return result.lower().strip()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE TYPE KEYWORDS
# ══════════════════════════════════════════════════════════════════════════════

TABLE_TYPE_KEYWORDS = {
    "mechanical_properties": {
        "keywords": [
            "uts", "ultimate tensile", "tensile strength",
            "yield strength", "yield stress", "proof stress",
            "elongation", "ductility", "hardness",
            "vickers", "rockwell", "brinell",
            "hv", "hrc", "hrb", "hb",
            "fatigue", "endurance limit", "fracture toughness",
            "elastic modulus", "young modulus", "stiffness",
            "impact energy", "charpy", "izod",
            "compression", "shear strength", "flexural",
            "sigma uts", "sigma y", "sigma u", "sigma 0.2",
            "sigma yield", "sigma ultimate",
            "epsilon f", "epsilon uts",
            "mpa", "gpa", "j/mm", "kj/m",
            "rp0.2", "rp 0.2", "rm ", "a50", "a5",
        ],
        "min_matches": 2,
    },
    "composition": {
        "keywords": [
            "wt%", "wt.%", "weight%", "weight percent",
            "at%", "at.%", "atomic%", "atomic percent",
            "mass fraction", "mole fraction",
            "nominal composition", "chemical composition",
            "alloy composition", "chemical analysis",
            "balance", "remainder", "bal.",
            "composition wt", "composition at",
            "alloying element", "trace element",
        ],
        "min_matches": 2,
    },
    "processing_parameters": {
        "keywords": [
            "solution treatment", "solution temperature",
            "aging temperature", "aging time", "ageing",
            "annealing temperature", "annealing time",
            "quenching", "quench rate", "cooling rate",
            "heating rate", "holding time",
            "extrusion ratio", "extrusion temperature",
            "rolling reduction", "forging temperature",
            "ecap", "hpt", "arb",
            "sintering temperature", "sintering time",
            "heat treatment", "temper", "t6", "t4", "t73",
            "strain rate", "deformation temperature",
            "process parameter", "processing condition",
        ],
        "min_matches": 2,
    },
    "microstructure": {
        "keywords": [
            "grain size", "grain diameter", "grain boundary",
            "phase fraction", "volume fraction", "area fraction",
            "precipitate size", "precipitate spacing",
            "dislocation density", "dislocation",
            "texture", "misorientation", "orientation",
            "aspect ratio", "interlamellar spacing",
            "crystallite size", "coherence length",
            "martensite fraction", "austenite fraction",
            "ferrite fraction", "pearlite fraction",
            "secondary phase", "intermetallic",
            "particle size", "particle distribution",
            "microhardness map", "ebsd", "xrd peak",
        ],
        "min_matches": 2,
    },
    "physical_properties": {
        "keywords": [
            "thermal conductivity", "heat conductivity",
            "electrical conductivity", "electrical resistivity",
            "coefficient of thermal expansion", "cte",
            "thermal expansion", "specific heat",
            "melting point", "melting temperature",
            "liquidus", "solidus",
            "density g/cm", "specific gravity",
            "diffusivity", "diffusion coefficient",
            "magnetic", "saturation magnetization",
        ],
        "min_matches": 2,
    },
    "corrosion": {
        "keywords": [
            "corrosion rate", "corrosion resistance",
            "weight loss", "mass loss",
            "pitting potential", "corrosion potential",
            "corrosion current", "icorr", "ecorr",
            "polarization curve", "tafel",
            "salt spray", "immersion test",
            "oxidation rate", "oxide thickness",
            "mm/year", "mpy", "mg/cm2",
            "eis", "impedance", "nyquist",
        ],
        "min_matches": 2,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PROPERTY EXTRACTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

PROPERTY_PATTERNS = {
    "UTS_MPa": [
        r"(?:UTS|ultimate\s*tensile\s*strength|tensile\s*strength|"
        r"[Ss]igma[_\s]*[Uu]|σ[_\s]*[Uu]|σ[_\s]*UTS)"
        r"[^\d]{0,25}(\d{2,4}(?:\.\d)?)\s*(?:MPa|mpa)",
        r"(\d{2,4}(?:\.\d)?)\s*MPa[^\d]{0,20}"
        r"(?:UTS|ultimate|tensile\s*strength)",
    ],
    "YS_MPa": [
        r"(?:YS|yield\s*strength|yield\s*stress|0\.2\s*%?\s*proof|"
        r"proof\s*stress|[Ss]igma[_\s]*[Yy]|σ[_\s]*[Yy]|"
        r"[Ss]igma[_\s]*0\.2|σ[_\s]*0\.2|Rp0?\.?2)"
        r"[^\d]{0,25}(\d{2,4}(?:\.\d)?)\s*(?:MPa|mpa)",
        r"(\d{2,4}(?:\.\d)?)\s*MPa[^\d]{0,20}(?:yield|YS|Rp)",
    ],
    "elongation_pct": [
        r"(?:elongation|ductility|El\.?|A\d*\s*%?|"
        r"[Ee]psilon[_\s]*[Ff]|ε[_\s]*[Ff])"
        r"[^\d]{0,20}(\d{1,3}(?:\.\d{1,2})?)\s*%",
        r"(\d{1,3}(?:\.\d{1,2})?)\s*%\s*(?:elongation|ductility)",
    ],
    "hardness_HV": [
        r"(\d{2,4}(?:\.\d)?)\s*(?:HV\d*|Vickers)",
        r"(?:[Hh]ardness|HV)[^\d]{0,20}(\d{2,4}(?:\.\d)?)",
    ],
    "hardness_HRC": [
        r"(\d{1,3}(?:\.\d)?)\s*HRC",
        r"(?:Rockwell|HRC)[^\d]{0,15}(\d{1,3}(?:\.\d)?)",
    ],
    "hardness_HB": [
        r"(\d{2,4}(?:\.\d)?)\s*(?:HB|Brinell)",
    ],
    "fatigue_MPa": [
        r"(?:fatigue\s*(?:limit|strength)|endurance\s*limit)"
        r"[^\d]{0,25}(\d{2,4}(?:\.\d)?)\s*MPa",
    ],
    "fracture_toughness": [
        r"(?:K[Ii][Cc]|fracture\s*toughness)"
        r"[^\d]{0,20}(\d{1,3}(?:\.\d)?)\s*MPa",
    ],
    "elastic_modulus_GPa": [
        r"(?:[Ee]lastic\s*modulus|[Yy]oung'?s?\s*modulus)"
        r"[^\d]{0,20}(\d{2,4}(?:\.\d)?)\s*GPa",
    ],
    "grain_size_um": [
        r"(?:grain\s*size|grain\s*diameter)"
        r"[^\d]{0,20}(\d{1,4}(?:\.\d)?)\s*(?:µm|um|μm|micron)",
    ],
    "density_g_cm3": [
        r"(?:density)[^\d]{0,20}"
        r"(\d{1,2}(?:\.\d{1,3})?)\s*g/cm",
    ],
    "corrosion_rate": [
        r"(?:corrosion\s*rate)[^\d]{0,20}"
        r"(\d{1,4}(?:\.\d{1,3})?)\s*(?:mm/year|mpy|µm/year)",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# ALLOY DESIGNATION PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

ALLOY_PATTERNS = {
    "Al": [
        r"\b(?:AA\s*)?(?:[1-8]\d{3})(?:-[A-Z]\d*(?:\d+)?)?\b",
        r"\bAlSi\d+(?:Mg|Cu)?\b",
        r"\bAl-\d+(?:\.\d+)?(?:Si|Zn|Mg|Cu|Li)\b",
    ],
    "Ti": [
        r"\bTi-\d+Al-\d+V\b",
        r"\bGrade\s*[1-5]\b",
        r"\bTi-\d+(?:Nb|Mo|Zr|V|Al)\b",
        r"\bTNZT\b",
    ],
    "Steel": [
        r"\b(?:AISI|SAE)\s*\d{4}[A-Z]?\b",
        r"\b(?:DP|CP|TRIP|TWIP|MS)\s*\d{3,4}\b",
        r"\b316L?\b|\b304L?\b|\b410\b|\b420\b|\b440[ABC]?\b",
        r"\bS\d{3}(?:ML|NL)?\b",
    ],
    "Mg": [
        r"\b(?:AZ|AM|ZK|ZE|WE|QE)\d+[A-Z]?\b",
        r"\bMg-\d+(?:\.\d+)?(?:Al|Zn|Ca|Gd|Y|Nd)\b",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION MARKERS
# ══════════════════════════════════════════════════════════════════════════════

SECTION_MARKERS = [
    "abstract", "introduction", "background",
    "experimental", "methodology", "methods",
    "materials and methods", "experimental procedure",
    "results", "discussion", "results and discussion",
    "conclusion", "conclusions", "summary",
    "acknowledgment", "acknowledgements", "references",
]


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts."""
    text = re.sub(r'-\n(?=[a-z])', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = text.replace('\x0c', '\n').replace('\x00', '')
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract full text from PDF using pdfminer."""
    try:
        text = pdfminer_extract(pdf_path)
        text = clean_text(text)
        return text if len(text) > 200 else None
    except (PDFSyntaxError, Exception) as e:
        logger.warning(f"pdfminer failed for {pdf_path}: {e}")
        return None


def split_into_sections(text: str) -> dict[str, str]:
    """Split paper text into named sections."""
    sections:      dict[str, str] = {}
    current_section                = "preamble"
    current_lines: list[str]       = []

    for line in text.split('\n'):
        stripped       = line.strip().lower()
        stripped_clean = re.sub(r'[^a-z\s]', '', stripped).strip()
        matched        = None

        for marker in SECTION_MARKERS:
            if (stripped_clean == marker or
                    (stripped_clean.startswith(marker)
                     and len(stripped) < 65)):
                matched = marker
                break

        if matched:
            if current_lines:
                sections[current_section] = '\n'.join(current_lines).strip()
            current_section = matched
            current_lines   = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = '\n'.join(current_lines).strip()

    return sections


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4: FUZZY CAPTION MATCHING BY CONTENT PROXIMITY
# Replaces index-based caption lookup which drifts when tables are skipped
# ══════════════════════════════════════════════════════════════════════════════

def extract_table_captions(full_text: str) -> dict[int, str]:
    """Extract all table captions from paper text by index."""
    captions = {}
    pattern  = re.compile(
        r'(?:Table|TABLE)\s*(\d+)[.\s:–-]*([^\n]{10,250})',
        re.IGNORECASE
    )
    for match in pattern.finditer(full_text):
        try:
            num     = int(match.group(1))
            caption = match.group(2).strip()
            if num not in captions:
                captions[num] = caption
        except (ValueError, IndexError):
            pass
    return captions


def find_best_caption(
    header:    list[str],
    rows:      list[dict],
    all_captions: dict[int, str],
) -> str:
    """
    IMPROVEMENT 4: Find best matching caption by content overlap
    rather than strict table index number.

    Why: pdfplumber's table counter drifts when it skips failed tables,
    so table 3 in the paper may be the 2nd table pdfplumber extracts.
    Content-based matching is more reliable.
    """
    if not all_captions:
        return ""

    # Build search words from header + first row values
    header_text = normalize_text(" ".join(str(h) for h in header))
    row_text    = ""
    if rows:
        row_text = normalize_text(
            " ".join(str(v) for v in rows[0].values())
        )
    combined_words = set(
        w for w in (header_text + " " + row_text).split()
        if len(w) > 2  # skip short words
    )

    best_caption = ""
    best_overlap = 0

    for num, caption in all_captions.items():
        caption_words = set(
            w for w in normalize_text(caption).split()
            if len(w) > 2
        )
        overlap = len(combined_words & caption_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_caption = caption

    # Only return caption if at least 1 meaningful word overlaps
    return best_caption if best_overlap >= 1 else ""


# ══════════════════════════════════════════════════════════════════════════════
# TABLE TYPE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_table_type(
    header:  list[str],
    rows:    list[dict],
    caption: str = "",
) -> tuple[str, float]:
    """
    Detect table type with three-layer detection:
    Layer 1 — keyword matching on header + caption + rows
    Layer 2 — unit-based detection for Greek-header tables
    Layer 3 — pattern detection in cell values

    Returns (table_type, confidence).
    """
    # Build searchable text
    raw_text  = " ".join(str(h) for h in header)
    raw_text += " " + caption
    for row in rows[:3]:
        raw_text += " " + " ".join(str(v) for v in row.values())
    search_text = normalize_text(raw_text)

    # Layer 1 — keyword matching
    best_type  = "general"
    best_score = 0

    for table_type, config in TABLE_TYPE_KEYWORDS.items():
        keywords    = config["keywords"]
        min_matches = config["min_matches"]
        score       = sum(1 for kw in keywords if kw in search_text)

        if score >= min_matches and score > best_score:
            best_score = score
            best_type  = table_type

    confidence = min(best_score / 4.0, 1.0)

    # Layer 2 — IMPROVEMENT 3: Unit-based detection for Greek-header tables
    # Many materials science tables have only Greek symbols as column headers
    # like σ_y | σ_UTS | ε_f | HV — keyword layer misses these
    if best_type == "general":
        # Check cell values for strong type indicators
        cell_sample = ""
        for row in rows[:6]:
            cell_sample += " ".join(str(v) for v in row.values()) + " "
        cell_norm = normalize_text(cell_sample)

        # MPa values → almost certainly mechanical properties
        if re.search(r'\b\d{2,4}\s*mpa\b', cell_norm):
            best_type  = "mechanical_properties"
            confidence = 0.6
            best_score = 3  # treat as moderate confidence

        # GPa values → elastic modulus / mechanical
        elif re.search(r'\b\d{2,3}\s*gpa\b', cell_norm):
            best_type  = "mechanical_properties"
            confidence = 0.5
            best_score = 2

        # wt% or at% patterns → composition
        elif re.search(
            r'\b\d{1,3}(?:\.\d{1,2})?\s*(?:wt|at)\.?%',
            cell_norm
        ):
            best_type  = "composition"
            confidence = 0.5
            best_score = 2

        # Temperature + time → processing parameters
        elif (re.search(r'\b\d{2,4}\s*°?\s*c\b', cell_norm) and
              re.search(r'\b\d+\s*(?:h|hr|min)\b', cell_norm)):
            best_type  = "processing_parameters"
            confidence = 0.5
            best_score = 2

        # µm values → microstructure (grain size etc.)
        elif re.search(r'\b\d+(?:\.\d+)?\s*(?:µm|um|nm)\b', cell_norm):
            best_type  = "microstructure"
            confidence = 0.4
            best_score = 2

        # mm/year or mpy → corrosion
        elif re.search(r'\b\d+(?:\.\d+)?\s*(?:mm/year|mpy)\b', cell_norm):
            best_type  = "corrosion"
            confidence = 0.5
            best_score = 2

    # Layer 3 — check header for Greek-normalized property names
    # e.g. header "σ_y" normalizes to "sigma y" which is in keywords
    if best_type == "general":
        header_norm = normalize_text(" ".join(str(h) for h in header))
        mech_greek  = [
            "sigma y", "sigma u", "sigma uts", "sigma 0.2",
            "epsilon f", "epsilon b", "sigma yield",
        ]
        if any(g in header_norm for g in mech_greek):
            best_type  = "mechanical_properties"
            confidence = 0.7
            best_score = 3

    return best_type, confidence


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5: TABLE QUALITY FILTER WITH NUMERICAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def is_valuable_table(
    header:     list[str],
    rows:       list[dict],
    table_type: str,
    confidence: float,
) -> bool:
    """
    IMPROVEMENT 5: Conservative filter with numerical data verification.
    Accepts typed tables at lower confidence (0.2) but verifies they
    actually contain numerical data — prevents confident misclassification.
    """
    # Structural minimums
    if len(header) < 2:
        return False
    if len(rows) < 2:
        return False

    # First row must have at least 2 non-empty cells
    first_row_values = [
        v for v in rows[0].values()
        if v and str(v).strip() and str(v).strip() not in ["-", "—", "/"]
    ]
    if len(first_row_values) < 2:
        return False

    # Skip reference/bibliography/author tables
    header_text  = " ".join(str(h) for h in header).lower()
    skip_phrases = [
        "reference", "author", "doi", "citation",
        "bibliography", "affiliation", "email",
        "funding", "grant number", "institution",
        "acknowledgment", "conflict",
    ]
    if any(phrase in header_text for phrase in skip_phrases):
        return False

    # Check numerical content across all rows (not just first)
    all_values = []
    for row in rows[:8]:
        all_values.extend(str(v) for v in row.values())
    all_text    = " ".join(all_values)
    has_numbers = bool(re.search(r'\d+(?:\.\d+)?', all_text))

    if not has_numbers:
        return False

    # IMPROVEMENT 5 KEY CHANGE:
    # Typed tables: lower confidence threshold (0.2 vs old 0.3)
    # but require numerical data (already verified above)
    if table_type != "general" and confidence >= 0.2:
        return True

    # General tables: require at least 3 rows
    if table_type == "general" and len(rows) >= 3:
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2: MULTI-PAGE TABLE MERGING
# ══════════════════════════════════════════════════════════════════════════════

def merge_continuation_tables(tables: list[dict]) -> list[dict]:
    """
    IMPROVEMENT 2: Detect and merge tables that span multiple pages.

    A table continued on the next page has:
    - Same or very similar header
    - Appears on the immediately following page

    Without this, a 15-row table split across pages 3-4 becomes
    two separate 7-row fragments, both potentially losing context.
    """
    if len(tables) <= 1:
        return tables

    merged = []
    i      = 0

    while i < len(tables):
        current = tables[i]

        if i + 1 < len(tables):
            next_t = tables[i + 1]

            # Check if next table is a continuation
            current_page = current.get("page", 0)
            next_page    = next_t.get("page", 0)
            pages_apart  = next_page - current_page

            # Must be on immediately following page (or same page, rare)
            if pages_apart <= 1:
                current_header = [
                    normalize_text(str(h)) for h in current.get("header", [])
                ]
                next_header    = [
                    normalize_text(str(h)) for h in next_t.get("header", [])
                ]

                # Headers match if same length and >70% overlap
                headers_match = False
                if len(current_header) == len(next_header):
                    matches = sum(
                        1 for a, b in zip(current_header, next_header)
                        if a == b or (a and b and a[:4] == b[:4])
                    )
                    headers_match = (
                        matches / max(len(current_header), 1) >= 0.7
                    )

                if headers_match:
                    # Merge: add next table's rows to current
                    current["rows"].extend(next_t.get("rows", []))
                    current["n_rows"] = len(current["rows"])
                    current["merged_from_pages"] = [
                        current_page, next_page
                    ]

                    # Rebuild text representation after merge
                    current["text_representation"] = (
                        build_table_text_representation(
                            current,
                            current.get("paper_title", ""),
                            current.get("alloy_system", "Unknown"),
                            current.get("alloy_desigs", []),
                        )
                    )

                    logger.debug(
                        f"Merged continuation table: "
                        f"page {current_page} + page {next_page}"
                    )
                    i += 2  # skip next table since it was merged
                    merged.append(current)
                    continue

        merged.append(current)
        i += 1

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# TABLE TEXT REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════

def build_table_text_representation(
    table_dict:   dict,
    paper_title:  str,
    alloy_system: str,
    alloy_desigs: list[str],
) -> str:
    """
    Build rich natural language text representation for embedding.
    Designed to be findable by both semantic and keyword search.
    """
    table_type = table_dict.get("table_type", "general")
    header     = table_dict.get("header", [])
    rows       = table_dict.get("rows", [])
    caption    = table_dict.get("caption", "")

    parts = []

    # Context header
    type_label = table_type.replace("_", " ")
    parts.append(
        f"Materials science {type_label} data table. "
        f"Alloy system: {alloy_system}."
    )

    if alloy_desigs:
        parts.append(
            f"Alloy designations mentioned: {', '.join(alloy_desigs[:8])}."
        )

    if caption:
        parts.append(f"Table caption: {caption[:200]}.")

    parts.append(f"From paper: {paper_title[:120]}.")

    # Column headers — normalized for embedding
    clean_headers = [
        normalize_text(str(h)) for h in header
        if h and str(h).strip()
    ]
    if clean_headers:
        parts.append(f"Table columns: {', '.join(clean_headers)}.")

    # Row data in natural language
    row_texts = []
    for row in rows[:25]:
        cell_parts = []
        for k, v in row.items():
            if not k or not v:
                continue
            k_norm = normalize_text(str(k))
            v_str  = str(v).strip()
            if (k_norm and v_str
                    and v_str not in ["-", "—", "N/A", "n/a", "/", ""]):
                cell_parts.append(f"{k_norm}: {v_str}")
        if len(cell_parts) >= 2:
            row_texts.append("; ".join(cell_parts))

    if row_texts:
        parts.append("Data: " + " || ".join(row_texts))

    # Property summary for mechanical tables
    if table_type == "mechanical_properties":
        summary = extract_property_summary_from_rows(rows, header)
        if summary:
            parts.append(f"Property values: {summary}")

    return " ".join(parts)


def extract_property_summary_from_rows(
    rows:   list[dict],
    header: list[str],
) -> str:
    """Extract key mechanical property values as compact summary."""
    prop_aliases = {
        "uts":        ["uts", "ultimate", "tensile strength",
                       "sigma u", "sigma uts", "rm"],
        "ys":         ["ys", "yield", "proof", "sigma y",
                       "sigma 0.2", "rp0.2", "rp 0.2"],
        "elongation": ["elong", "el.", "a%", "a5", "a50",
                       "epsilon f", "ductility"],
        "hardness":   ["hv", "hrc", "hb", "hardness",
                       "vickers", "rockwell"],
        "fatigue":    ["fatigue", "endurance", "sigma -1"],
    }

    summaries = []
    for row in rows[:6]:
        row_props = []
        for col_name, val in row.items():
            if not col_name or not val:
                continue
            col_norm = normalize_text(str(col_name))
            val_str  = str(val).strip()
            for prop_key, aliases in prop_aliases.items():
                if any(alias in col_norm for alias in aliases):
                    if re.search(r'\d', val_str):
                        row_props.append(f"{prop_key}={val_str}")
                    break
        if row_props:
            summaries.append("; ".join(row_props))

    return " || ".join(summaries[:5])


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1: THREE-STRATEGY TABLE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_tables_from_pdf(
    pdf_path:     str,
    paper_title:  str       = "",
    alloy_system: str       = "Unknown",
    alloy_desigs: list[str] = None,
    full_text:    str       = "",
) -> list[dict]:
    """
    IMPROVEMENT 1: Three-strategy table extraction.

    Strategy 1: lines_strict — bordered tables (most precise)
    Strategy 2: lines        — semi-bordered tables
    Strategy 3: text         — NEW: borderless/whitespace tables
                               common in materials science papers
                               that use whitespace alignment

    All three strategies try in sequence per page.
    First strategy that finds tables wins for that page.
    """
    if alloy_desigs is None:
        alloy_desigs = []

    tables      = []
    table_index = 0

    # Pre-extract all captions for fuzzy matching (Improvement 4)
    all_captions = extract_table_captions(full_text) if full_text else {}

    # Three strategies — ordered from most to least precise
    extraction_strategies = [
        {
            "name": "lines_strict",
            "settings": {
                "vertical_strategy":   "lines_strict",
                "horizontal_strategy": "lines_strict",
                "snap_tolerance":       4,
                "join_tolerance":       3,
                "edge_min_length":     15,
                "min_words_vertical":   1,
                "min_words_horizontal": 1,
            }
        },
        {
            "name": "lines",
            "settings": {
                "vertical_strategy":   "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance":       5,
                "join_tolerance":       4,
            }
        },
        {
            # IMPROVEMENT 1: text strategy for borderless tables
            "name": "text",
            "settings": {
                "vertical_strategy":   "text",
                "horizontal_strategy": "lines",
                "snap_tolerance":       8,
                "intersection_tolerance": 8,
                "text_tolerance":        4,
            }
        },
    ]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):

                page_tables = []

                # Try strategies in order until one finds tables
                for strategy in extraction_strategies:
                    try:
                        candidate = page.extract_tables(
                            strategy["settings"]
                        )
                        if candidate:
                            page_tables = candidate
                            break  # use first strategy that works
                    except Exception:
                        continue

                for raw_table in page_tables:
                    if not raw_table or len(raw_table) < 3:
                        table_index += 1
                        continue

                    # Parse header
                    raw_header = raw_table[0]
                    header     = []
                    for cell in raw_header:
                        cell_str = str(cell).strip() if cell else ""
                        cell_str = re.sub(r'[\x00-\x1f\x7f]', ' ', cell_str)
                        cell_str = re.sub(r'\s+', ' ', cell_str).strip()
                        header.append(cell_str)

                    non_empty_headers = [h for h in header if h.strip()]
                    if len(non_empty_headers) < 2:
                        table_index += 1
                        continue

                    # Parse rows
                    rows = []
                    for raw_row in raw_table[1:]:
                        row_dict = {}
                        for col_idx, cell in enumerate(raw_row):
                            if col_idx >= len(header):
                                break
                            col_name = header[col_idx] or f"col_{col_idx}"
                            val      = str(cell).strip() if cell else ""
                            val      = re.sub(r'[\x00-\x1f\x7f]', ' ', val)
                            val      = re.sub(r'\s+', ' ', val).strip()
                            row_dict[col_name] = val

                        non_empty = [
                            v for v in row_dict.values()
                            if v and v not in ["-", "—", "/", ""]
                        ]
                        if len(non_empty) >= 2:
                            rows.append(row_dict)

                    if len(rows) < 2:
                        table_index += 1
                        continue

                    # IMPROVEMENT 4: Fuzzy caption matching
                    caption = find_best_caption(header, rows, all_captions)

                    # Detect type (includes Improvement 3)
                    table_type, confidence = detect_table_type(
                        header, rows, caption
                    )

                    # IMPROVEMENT 5: Quality filter with numerical check
                    if not is_valuable_table(
                        header, rows, table_type, confidence
                    ):
                        table_index += 1
                        continue

                    table_dict = {
                        "table_id":        f"table_{table_index}",
                        "page":            page_num + 1,
                        "table_number":    table_index + 1,
                        "caption":         caption,
                        "header":          header,
                        "rows":            rows,
                        "table_type":      table_type,
                        "type_confidence": round(confidence, 3),
                        "alloy_system":    alloy_system,
                        "alloy_desigs":    alloy_desigs,
                        "paper_title":     paper_title,
                        "n_rows":          len(rows),
                        "n_cols":          len(header),
                    }

                    table_dict["text_representation"] = (
                        build_table_text_representation(
                            table_dict,
                            paper_title,
                            alloy_system,
                            alloy_desigs,
                        )
                    )

                    tables.append(table_dict)
                    table_index += 1

    except Exception as e:
        logger.warning(f"Table extraction failed for {pdf_path}: {e}")

    # IMPROVEMENT 2: Merge multi-page continuation tables
    if len(tables) > 1:
        tables_before = len(tables)
        tables        = merge_continuation_tables(tables)
        if len(tables) < tables_before:
            logger.debug(
                f"Merged {tables_before - len(tables)} "
                f"continuation table(s)"
            )

    return tables


# ══════════════════════════════════════════════════════════════════════════════
# PROPERTY EXTRACTION FROM TEXT
# ══════════════════════════════════════════════════════════════════════════════

def extract_properties_from_text(text: str) -> dict[str, list[float]]:
    """Extract numerical property values from free text."""
    normalized = normalize_text(text)
    found: dict[str, list[float]] = {}

    for prop_name, patterns in PROPERTY_PATTERNS.items():
        values = []
        for pattern in patterns:
            for search_text in [text, normalized]:
                matches = re.findall(pattern, search_text, re.IGNORECASE)
                for m in matches:
                    try:
                        val = float(str(m).replace(',', '.'))
                        if val > 0:
                            values.append(val)
                    except (ValueError, TypeError):
                        pass
        if values:
            found[prop_name] = sorted(list(set(values)))

    return found


def extract_alloy_designations(text: str, alloy_system: str) -> list[str]:
    """Extract alloy designations from text."""
    patterns = ALLOY_PATTERNS.get(alloy_system, [])
    if not patterns:
        patterns = [p for plist in ALLOY_PATTERNS.values() for p in plist]

    found = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = m.strip().upper()
            if len(cleaned) >= 3:
                found.add(cleaned)

    return sorted(list(found))


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def create_chunks(
    sections:   dict[str, str],
    paper_id:   str,
    chunk_size: int = 400,
    overlap:    int = 60,
) -> list[dict]:
    """Split sections into overlapping word chunks for retrieval."""
    chunks    = []
    chunk_idx = 0

    skip_sections  = {
        "references", "bibliography",
        "acknowledgment", "acknowledgements"
    }
    priority_order = [
        "results and discussion", "results", "discussion",
        "experimental", "methodology", "methods",
        "abstract", "introduction", "conclusion", "conclusions",
        "preamble",
    ]

    sorted_sections = sorted(
        sections.items(),
        key=lambda x: next(
            (i for i, s in enumerate(priority_order)
             if s in x[0].lower()),
            len(priority_order)
        )
    )

    for section_name, section_text in sorted_sections:
        if not section_text or len(section_text) < 50:
            continue
        if any(skip in section_name.lower() for skip in skip_sections):
            continue

        words = section_text.split()
        start = 0

        while start < len(words):
            end        = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            if len(chunk_text) > 100:
                chunks.append({
                    "chunk_id":   f"{paper_id}_chunk_{chunk_idx}",
                    "paper_id":   paper_id,
                    "section":    section_name,
                    "text":       chunk_text,
                    "word_count": end - start,
                })
                chunk_idx += 1

            if end >= len(words):
                break
            start = end - overlap

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE PAPER PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_paper(
    metadata: dict,
    pdf_path: Optional[str] = None,
) -> dict:
    """Full extraction pipeline for one paper."""
    paper_id     = metadata["paper_id"]
    alloy_system = metadata.get("alloy_system", "Unknown")
    title        = metadata.get("title", "")

    full_text:    Optional[str]  = None
    sections:     dict[str, str] = {}
    tables:       list[dict]     = []
    alloy_desigs: list[str]      = []

    if pdf_path and Path(pdf_path).exists():
        full_text = extract_text_from_pdf(pdf_path)

        if full_text:
            sections     = split_into_sections(full_text)
            alloy_desigs = extract_alloy_designations(full_text, alloy_system)
            tables       = extract_tables_from_pdf(
                pdf_path,
                paper_title=title,
                alloy_system=alloy_system,
                alloy_desigs=alloy_desigs,
                full_text=full_text,
            )

    if not sections:
        abstract = metadata.get("abstract", "")
        if abstract:
            sections     = {"abstract": abstract}
            alloy_desigs = extract_alloy_designations(abstract, alloy_system)

    chunks           = create_chunks(sections, paper_id)
    combined_text    = full_text or metadata.get("abstract", "")
    properties_found = extract_properties_from_text(combined_text)

    for t in tables:
        t["table_id"] = f"{paper_id}_{t['table_id']}"
        t["paper_id"] = paper_id

    type_counts: dict[str, int] = {}
    for t in tables:
        ttype = t.get("table_type", "general")
        type_counts[ttype] = type_counts.get(ttype, 0) + 1

    return {
        "paper_id":             paper_id,
        "title":                title,
        "authors":              metadata.get("authors", []),
        "year":                 metadata.get("year"),
        "doi":                  metadata.get("doi"),
        "journal":              metadata.get("journal"),
        "alloy_system":         alloy_system,
        "alloy_designations":   alloy_desigs,
        "properties_extracted": properties_found,
        "has_full_text":        full_text is not None,
        "sections":             sections,
        "text_chunks":          chunks,
        "property_tables":      tables,
        "chunk_count":          len(chunks),
        "table_count":          len(tables),
        "table_type_counts":    type_counts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_all_papers(
    metadata_path:  str  = "data/raw_metadata/papers_metadata.json",
    pdf_map_path:   str  = "data/raw_metadata/pdf_map.json",
    output_dir:     str  = "data/corpus",
    skip_existing:  bool = True,
) -> None:
    """
    Process all collected papers and save to data/corpus/.
    skip_existing=True skips already-processed papers for fast incremental runs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not Path(metadata_path).exists():
        logger.error(f"Metadata not found at: {metadata_path}")
        return

    with open(metadata_path, encoding="utf-8") as f:
        all_metadata = json.load(f)

    pdf_map = {}
    if Path(pdf_map_path).exists():
        with open(pdf_map_path, encoding="utf-8") as f:
            pdf_map = json.load(f)

    # Skip already processed papers
    if skip_existing:
        existing_ids   = {p.stem for p in Path(output_dir).glob("*.json")}
        original_count = len(all_metadata)
        all_metadata   = [
            m for m in all_metadata
            if m["paper_id"] not in existing_ids
        ]
        skipped = original_count - len(all_metadata)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed papers")
        if not all_metadata:
            logger.info("All papers already processed — nothing to do")
            logger.info("To reprocess all: run with skip_existing=False")
            return

    logger.info(f"Processing {len(all_metadata)} papers...")

    stats = {
        "total":         0,
        "with_pdf":      0,
        "abstract_only": 0,
        "total_chunks":  0,
        "total_tables":  0,
        "table_types":   {},
    }

    for i, meta in enumerate(all_metadata):
        pid      = meta["paper_id"]
        pdf_path = pdf_map.get(pid)

        try:
            doc = process_paper(meta, pdf_path)
        except Exception as e:
            logger.warning(f"Failed to process {pid}: {e}")
            continue

        out_path = Path(output_dir) / f"{pid}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        stats["total"]        += 1
        stats["total_chunks"] += doc["chunk_count"]
        stats["total_tables"] += doc["table_count"]

        for ttype, count in doc.get("table_type_counts", {}).items():
            stats["table_types"][ttype] = (
                stats["table_types"].get(ttype, 0) + count
            )

        if doc["has_full_text"]:
            stats["with_pdf"] += 1
        else:
            stats["abstract_only"] += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(all_metadata):
            logger.info(
                f"Processed {i+1}/{len(all_metadata)} | "
                f"chunks: {stats['total_chunks']} | "
                f"tables: {stats['total_tables']}"
            )

    logger.info("")
    logger.info("=" * 55)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Total papers:       {stats['total']}")
    logger.info(f"  With full PDF text: {stats['with_pdf']}")
    logger.info(f"  Abstract only:      {stats['abstract_only']}")
    logger.info(f"  Total text chunks:  {stats['total_chunks']}")
    logger.info(f"  Total tables:       {stats['total_tables']}")
    logger.info(
        f"  Avg tables/paper:   "
        f"{stats['total_tables']/max(stats['with_pdf'],1):.1f}"
    )
    logger.info("")
    logger.info("  Table types:")
    for ttype, count in sorted(
        stats["table_types"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        logger.info(f"    {ttype:<30} {count}")
    logger.info("=" * 55)


if __name__ == "__main__":
    process_all_papers()