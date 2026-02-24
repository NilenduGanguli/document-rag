"""
NLP service for entity extraction.

Two-pass approach:
  1. spaCy en_core_web_lg — fast, high-quality NER for standard entity types
     (PERSON, ORG, DATE, GPE/ADDRESS) plus a custom regex pass for PASSPORT_NUM.
  2. GLiNER (optional) — zero-shot NER for KYC-specific entity types that
     spaCy's general vocabulary does not cover: UBO, Beneficial Owner,
     Company Registration Number, Tax ID, IBAN, Sanction Entry.

Each returned entity dict has:
  { "type": str, "value": str, "confidence": float, "method": str }

Cross-chunk context:
  `extract_entities_windowed()` runs NER over a sliding window of adjacent
  chunks so entities that span chunk boundaries are still detected.
"""
import logging
import re
from typing import List, Dict, Any

import spacy
import spacy.cli  # must be at module level to avoid UnboundLocalError in _load_spacy

logger = logging.getLogger(__name__)

# ── spaCy model (en_core_web_lg preferred; fall back to en_core_web_sm) ───────

def _load_spacy():
    for model_name in ("en_core_web_lg", "en_core_web_sm"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    # Last resort: download the small model
    spacy.cli.download("en_core_web_sm")
    return spacy.load("en_core_web_sm")

nlp = _load_spacy()
logger.info("Loaded spaCy model: %s", nlp.meta.get("name", "unknown"))

# ── GLiNER (optional, loaded lazily) ─────────────────────────────────────────

_gliner_model = None
_GLINER_MODEL_NAME = "urchade/gliner_mediumv2.1"
# KYC-specific labels not covered by spaCy's general NER
_GLINER_LABELS = [
    "UBO",
    "Beneficial Owner",
    "Company Registration Number",
    "Tax ID",
    "IBAN",
    "Sanction Entry",
]

def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        try:
            from gliner import GLiNER
            _gliner_model = GLiNER.from_pretrained(_GLINER_MODEL_NAME)
            logger.info("Loaded GLiNER model: %s", _GLINER_MODEL_NAME)
        except Exception as exc:
            logger.warning("GLiNER unavailable (%s). KYC-specific entity types will not be extracted.", exc)
            _gliner_model = False  # Sentinel: don't retry
    return _gliner_model if _gliner_model is not False else None

# ── Entity type mapping ───────────────────────────────────────────────────────

# spaCy label → domain EntityType string
_SPACY_TYPE_MAP: Dict[str, str] = {
    "PERSON":  "PERSON",
    "ORG":     "ORG",
    "DATE":    "DATE",
    "GPE":     "ADDRESS",
    "LOC":     "ADDRESS",
    "FAC":     "ADDRESS",
}

# GLiNER label → domain EntityType string
_GLINER_TYPE_MAP: Dict[str, str] = {
    "UBO":                          "UBO",
    "Beneficial Owner":             "BENEFICIAL_OWNER",
    "Company Registration Number":  "COMPANY_REG_NUM",
    "Tax ID":                       "TAX_ID",
    "IBAN":                         "IBAN",
    "Sanction Entry":               "SANCTION_ENTRY",
}

# Regex patterns for document-specific entities that neither model reliably catches
_PASSPORT_PATTERN = re.compile(r"\b[A-Z]{1,2}[0-9]{6,8}\b")
_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]{0,16})?\b")
_TAX_ID_PATTERN = re.compile(r"\b(?:TIN|EIN|TAX[-\s]?ID)[\s:]*([A-Z0-9\-]{5,20})\b", re.IGNORECASE)


# ── Main extraction function ──────────────────────────────────────────────────

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extracts entities from a single text string.

    Returns a list of dicts:
        {"type": str, "value": str, "confidence": float, "method": str}
    """
    seen: set = set()
    entities: List[Dict[str, Any]] = []

    def _add(etype: str, value: str, confidence: float, method: str):
        key = (etype, value.strip())
        if key not in seen:
            seen.add(key)
            entities.append({"type": etype, "value": value.strip(),
                              "confidence": confidence, "method": method})

    # Pass 1: spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        etype = _SPACY_TYPE_MAP.get(ent.label_)
        if etype:
            # spaCy does not expose per-entity confidence; use 0.85 as default
            _add(etype, ent.text, 0.85, "spacy")

    # Pass 2: GLiNER KYC-specific entities
    gliner = _get_gliner()
    if gliner is not None:
        try:
            predictions = gliner.predict_entities(text, _GLINER_LABELS, threshold=0.5)
            for pred in predictions:
                etype = _GLINER_TYPE_MAP.get(pred["label"])
                if etype:
                    _add(etype, pred["text"], float(pred.get("score", 0.75)), "gliner")
        except Exception as exc:
            logger.warning("GLiNER prediction failed: %s", exc)

    # Pass 3: Regex for structured document identifiers
    for match in _PASSPORT_PATTERN.finditer(text):
        _add("PASSPORT_NUM", match.group(0), 1.0, "regex")

    for match in _IBAN_PATTERN.finditer(text):
        _add("IBAN", match.group(0), 1.0, "regex")

    for match in _TAX_ID_PATTERN.finditer(text):
        value = match.group(1) if match.lastindex else match.group(0)
        _add("TAX_ID", value, 1.0, "regex")

    return entities


def extract_entities_windowed(chunks: List[str], window: int = 2) -> List[List[Dict[str, Any]]]:
    """
    Runs entity extraction over a sliding context window to catch entities
    that span adjacent chunk boundaries.

    For each chunk i, NER is run on the combined text of chunks [i-window, i+window].
    Only entities whose value overlaps with chunk i's own text are attributed to i.

    Returns a list of entity lists, one per input chunk.
    """
    results: List[List[Dict[str, Any]]] = []

    for i, chunk in enumerate(chunks):
        lo = max(0, i - window)
        hi = min(len(chunks), i + window + 1)
        context = " ".join(chunks[lo:hi])

        # Extract from the wider context
        context_entities = extract_entities(context)

        # Keep only entities whose value appears in the chunk itself
        chunk_entities = [e for e in context_entities if e["value"].lower() in chunk.lower()]
        results.append(chunk_entities)

    return results
