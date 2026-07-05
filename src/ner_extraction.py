"""
ner_extraction.py

Extracts structured entities from financial text chunks: organizations,
people (typically executives), monetary figures, dates, percentages.

WHY a Transformer-based NER pipeline instead of regex/rule-based extraction:
Financial text refers to entities inconsistently ("the Company," "management,"
"our CFO," varying monetary formats like "$1.2B" vs "1,200 million dollars").
A trained NER model generalizes across this variation; regex rules would need
constant expansion and still miss novel phrasings.

Uses a general-purpose Transformer NER model as the base. For a production
system, this would be fine-tuned on financial-domain-labeled data (financial
NER datasets exist, e.g. FiNER) to properly distinguish financial-specific
entity types (TICKER, MONETARY_VALUE) beyond generic PERSON/ORG/DATE.
"""
from dataclasses import dataclass
from typing import List
from transformers import pipeline


@dataclass
class ExtractedEntity:
    text: str
    label: str
    confidence: float


class FinancialNER:
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """
        dslim/bert-base-NER is a general-purpose fine-tuned BERT NER model
        (PERSON, ORG, LOC, MISC). It's used here as the base extractor since
        it's freely available and requires no gated access — swap for a
        finance-fine-tuned NER model (e.g. a FiNER-trained checkpoint) for
        production-grade financial entity typing.
        """
        self.pipe = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",  # merges sub-word tokens into whole entities
        )

    def extract(self, text: str) -> List[ExtractedEntity]:
        raw_entities = self.pipe(text)
        return [
            ExtractedEntity(
                text=e["word"],
                label=e["entity_group"],
                confidence=float(e["score"]),
            )
            for e in raw_entities
        ]

    def extract_batch(self, texts: List[str]) -> List[List[ExtractedEntity]]:
        return [self.extract(t) for t in texts]


import re

MONEY_PATTERN = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:million|billion|thousand|M|B|K)?", re.IGNORECASE
)
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s?%")


def extract_financial_figures(text: str) -> List[ExtractedEntity]:
    """
    Complements the Transformer NER pass with regex extraction specifically
    for monetary values and percentages. This is a deliberate hybrid design:
    general-purpose NER models are not reliably trained to tag monetary
    amounts and percentages as a distinct type, while regex is genuinely
    well-suited to these highly-structured, format-predictable patterns.
    This is standard practice — use the right tool per entity type rather
    than forcing one method to handle everything.
    """
    entities = []
    for match in MONEY_PATTERN.finditer(text):
        entities.append(ExtractedEntity(text=match.group(), label="MONETARY_VALUE", confidence=1.0))
    for match in PERCENT_PATTERN.finditer(text):
        entities.append(ExtractedEntity(text=match.group(), label="PERCENTAGE", confidence=1.0))
    return entities
