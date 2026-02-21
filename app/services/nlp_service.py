import spacy
import re
from typing import List, Dict

# Ensure the model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Extracts entities from text using spaCy and custom regex for KYC specific entities.
    """
    doc = nlp(text)
    entities = []
    
    # Extract standard entities
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "DATE", "GPE"]:
            entities.append({"type": ent.label_, "value": ent.text})
            
    # Custom regex for PASSPORT_NUM (e.g., A1234567, 123456789)
    # This is a simplified heuristic for demonstration
    passport_matches = re.finditer(r'\b[A-Z][0-9]{7}\b|\b[A-Z0-9]{5,9}\b', text)
    for match in passport_matches:
        val = match.group(0)
        # Simple heuristic to avoid matching random short words
        if any(char.isdigit() for char in val):
            # Check if we already added this to avoid duplicates
            if not any(e["value"] == val and e["type"] == "PASSPORT_NUM" for e in entities):
                entities.append({"type": "PASSPORT_NUM", "value": val})
                
    return entities
