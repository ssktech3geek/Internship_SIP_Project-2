"""
keyword_extraction.py

Extracts salient keyphrases per document chunk using KeyBERT.

WHY KeyBERT specifically: KeyBERT scores candidate n-gram phrases by their
embedding similarity to the full chunk's embedding, rather than relying on
frequency statistics (like TF-IDF). This matters for financial text because
the most *frequent* words in a filing are often boilerplate ("Company,"
"fiscal year"), while the most *meaningful* themes ("supply chain disruption,"
"margin compression") are exactly what KeyBERT's embedding-similarity
approach is designed to surface.
"""
from typing import List, Tuple
from keybert import KeyBERT


class ThemeExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # KeyBERT can use a lighter general-purpose encoder here even though
        # FinBERT is used for the main retrieval embeddings — keyword
        # extraction quality doesn't require the same domain-specificity
        # that whole-chunk semantic retrieval does, and MiniLM is faster.
        self.kw_model = KeyBERT(model=model_name)

    def extract_themes(
        self,
        text: str,
        ngram_range: Tuple[int, int] = (1, 2),
        top_n: int = 5,
        use_stopwords: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        ngram_range=(1,2): captures both single-word terms ("inflation")
        and short meaningful phrases ("supply chain") — single-word-only
        extraction misses most real financial themes, which are typically
        two-word compounds.

        Stopword removal prevents generic connector words from being
        scored as candidate phrases in the first place.
        """
        stop_words = "english" if use_stopwords else None
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words=stop_words,
            top_n=top_n,
        )
        return keywords
