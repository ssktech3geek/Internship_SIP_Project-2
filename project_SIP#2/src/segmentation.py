"""
segmentation.py

Splits long financial documents into meaningful, retrievable chunks.

WHY structure-aware segmentation instead of fixed-length windows:
Fixed-length chunking (e.g. "every 500 characters") can split a sentence,
a financial figure, or a risk disclosure mid-thought — which corrupts both
the embedding (it no longer represents a coherent idea) and any NER/keyword
extraction run on that chunk. Financial documents have natural structure
(sections in a 10-K, speaker turns in an earnings call transcript) that
should define chunk boundaries instead.
"""
import re
from dataclasses import dataclass
from typing import List


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_doc: str
    section: str = "unspecified"


def segment_by_paragraph(text: str, source_doc: str, min_chunk_length: int = 50) -> List[DocumentChunk]:
    """
    Split on paragraph boundaries (blank lines). This is the simplest
    structure-aware segmentation — appropriate when explicit section
    headers aren't reliably present (e.g. plain-text transcripts).

    min_chunk_length filters out near-empty fragments (stray blank lines,
    page numbers) that would otherwise pollute the embedding index with
    near-meaningless vectors.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        if len(para) < min_chunk_length:
            continue
        chunks.append(DocumentChunk(
            chunk_id=f"{source_doc}_p{i:03d}",
            text=para,
            source_doc=source_doc,
        ))
    return chunks


def segment_by_speaker(transcript: str, source_doc: str) -> List[DocumentChunk]:
    """
    Split an earnings call transcript by speaker turn.

    Expects lines in the format "Speaker Name: dialogue text".
    Speaker-turn segmentation matters specifically for transcripts because
    a single paragraph-based split can merge an analyst's question with
    the executive's answer, muddying both retrieval relevance and any
    per-speaker sentiment/entity analysis downstream.
    """
    pattern = re.compile(r"^([A-Z][A-Za-z\.\s]{2,40}):\s*(.+)$")
    chunks = []
    idx = 0
    for line in transcript.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            speaker, dialogue = match.groups()
            chunks.append(DocumentChunk(
                chunk_id=f"{source_doc}_s{idx:03d}",
                text=dialogue.strip(),
                source_doc=source_doc,
                section=speaker.strip(),
            ))
            idx += 1
    return chunks


def segment_by_section_headers(text: str, source_doc: str, headers: List[str]) -> List[DocumentChunk]:
    """
    Split a filing using known section headers (e.g. "Risk Factors",
    "Management's Discussion and Analysis").

    This is the most useful segmentation for SEC filings specifically,
    since it lets downstream retrieval be filtered by section
    (e.g. "only search within Risk Factors") — a real analyst workflow.
    """
    escaped = [re.escape(h) for h in headers]
    split_pattern = re.compile(r"(" + "|".join(escaped) + r")", re.IGNORECASE)
    parts = split_pattern.split(text)

    chunks = []
    current_section = "preamble"
    idx = 0
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue
        if any(part_stripped.lower() == h.lower() for h in headers):
            current_section = part_stripped
            continue
        chunks.append(DocumentChunk(
            chunk_id=f"{source_doc}_sec{idx:03d}",
            text=part_stripped,
            source_doc=source_doc,
            section=current_section,
        ))
        idx += 1
    return chunks
