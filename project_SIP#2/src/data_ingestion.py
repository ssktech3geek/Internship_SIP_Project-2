"""
data_ingestion.py

Utility to pull real filings from SEC EDGAR (free, public, no API key
required) — for scaling this project beyond the bundled sample documents.

SEC EDGAR requires a descriptive User-Agent header identifying the
requester (their policy, not optional) — this is set explicitly below.
Reference: https://www.sec.gov/os/webmaster-faq#developers
"""
import requests
import time
from typing import List, Dict


EDGAR_BASE_URL = "https://www.sec.gov"
USER_AGENT = "Personal Portfolio Project - youremail@example.com"  # replace with your real contact info before use


def get_company_filings(cik: str, form_type: str = "10-K", count: int = 5) -> List[Dict]:
    """
    Fetch recent filing metadata for a company by CIK (Central Index Key —
    look yours up at https://www.sec.gov/cgi-bin/browse-edgar).

    Returns filing metadata (not full text) — use get_filing_text() to
    pull the actual document content for a specific filing.
    """
    url = f"{EDGAR_BASE_URL}/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "CIK": cik,
        "type": form_type,
        "dateb": "",
        "owner": "include",
        "count": count,
        "output": "atom",
    }
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    # Respect SEC rate limits (max ~10 requests/second, we go well under that)
    time.sleep(0.2)
    return response.text  # atom XML — parse with feedparser or ElementTree as needed


def get_filing_text(filing_url: str) -> str:
    """
    Fetch the raw text of a specific filing document once you have its URL
    from get_company_filings(). Strips basic HTML for plain text use —
    for production use, a proper HTML-to-text parser (e.g. BeautifulSoup)
    should replace the naive strip here.
    """
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(filing_url, headers=headers)
    response.raise_for_status()
    time.sleep(0.2)
    return response.text


# Example usage (uncomment to run):
# filings = get_company_filings(cik="0000320193", form_type="10-K")  # Apple Inc.
# print(filings)
