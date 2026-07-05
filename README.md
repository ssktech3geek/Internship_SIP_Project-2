# Financial Document Intelligence System

**Author:** Siddharth Khedekar

A retrieval-augmented system for extracting structured insight from long financial
documents (SEC filings, earnings call transcripts) — segmentation, financial NER,
keyphrase extraction, domain-specific (FinBERT) embeddings, and FAISS-based
semantic retrieval, following the architecture pattern used in production
financial-document search tools (e.g., AlphaSense, Bloomberg document search).

## Problem

Financial analysts spend hours manually reading 10-K/10-Q filings and earnings
call transcripts to find risk disclosures, sentiment shifts, and specific
financial commentary. This project builds the retrieval + extraction backbone
that makes that process searchable and structured instead of manual.

## Architecture

```
Raw filing / transcript
        │
        ▼
 ┌──────────────┐   Split into meaningful, retrievable units
 │ Segmentation │   (by section / speaker turn / paragraph) —
 └──────┬───────┘   a 100+ page filing cannot be embedded as one blob
        │
        ▼
 ┌──────────────┐   Extract structured entities: companies, executives,
 │     NER      │   monetary figures, dates — enables entity-filtered
 └──────┬───────┘   search, not just free-text search
        │
        ▼
 ┌──────────────┐   Extract per-chunk themes (e.g. "supply chain",
 │   KeyBERT    │   "guidance cut", "inflation pressure") for fast
 └──────┬───────┘   thematic triage without reading full text
        │
        ▼
 ┌──────────────┐   Domain-specific Transformer embeddings —
 │   FinBERT    │   FinBERT outperforms generic sentence encoders
 └──────┬───────┘   on financial language specifically
        │
        ▼
 ┌──────────────┐   Fast approximate nearest-neighbor search
 │    FAISS     │   across all embedded chunks
 └──────┬───────┘
        │
        ▼
   Query → retrieve relevant chunks (+ entities + themes)
   → optionally synthesize a grounded answer via LLM
```

## Why these specific technical choices

| Component | Choice | Reasoning |
|---|---|---|
| Embedding model | `ProsusAI/finbert` | Pretrained specifically on financial text; generic sentence encoders (e.g. `all-MiniLM`) underperform on domain-specific financial language and terminology |
| Segmentation | Section/speaker-aware chunking, not fixed-length windows | Naive fixed-length chunking can split a sentence or a risk disclosure mid-thought; structure-aware segmentation preserves semantic units |
| Retrieval | FAISS (flat L2 index for this scale) | Standard, well-understood ANN library; flat index chosen deliberately at this corpus size — would move to IVF/HNSW at production scale (noted as a scaling consideration, not implemented for a corpus this small) |
| NER | Transformer-based NER pipeline | Rule-based/regex entity extraction fails on the linguistic variety of real financial text (e.g. "the Company," "management," inconsistent monetary formatting) |

## Project Structure

```
findoc/
├── README.md
├── requirements.txt
├── data/
│   └── sample_filings/       # small sample docs for demo (not real filings — see note below)
├── src/
│   ├── segmentation.py       # document chunking logic
│   ├── ner_extraction.py     # financial entity extraction
│   ├── keyword_extraction.py # KeyBERT theme extraction
│   ├── embedding_index.py    # FinBERT embedding + FAISS index/search
│   └── data_ingestion.py     # SEC EDGAR fetch utility (for real data)
└── notebooks/
    └── financial_document_intelligence.ipynb   # end-to-end walkthrough
```

## Data

This repo ships with small **synthetic sample documents** (not real filings) in
`data/sample_filings/` purely so the pipeline runs out of the box without
requiring any download. `src/data_ingestion.py` includes a utility to pull real
filings from **SEC EDGAR** (free, public, no API key required) to scale this up
to a real corpus.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebook: `notebooks/financial_document_intelligence.ipynb`

## Status

Prototype / portfolio project — demonstrates the full pipeline end-to-end on a
small sample corpus. Documented next steps (in the notebook) cover scaling to
a real EDGAR-sourced corpus and swapping the flat FAISS index for an ANN index
at larger scale.
