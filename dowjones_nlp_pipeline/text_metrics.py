import os
import re
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import lru_cache
from config import SEC_FILINGS_PATH

import time

# Ensure required NLTK resources are available
for resource in ['punkt_tab', 'punkt', 'cmudict']:
    try:
        if resource in ['punkt_tab', 'punkt']:
            nltk.data.find(f'tokenizers/{resource}')
        else:
            nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def make_numsyllables(pronouncing_dict):
    """Return a cached syllable-counting function bound to a specific pronouncing_dict."""
    @lru_cache(maxsize=None)
    def _numsyllables(word):
        try:
            syll_list = [x for x in pronouncing_dict[word][0] if x[-1].isdigit()]
            return len(syll_list)
        except KeyError:
            return 0
    return _numsyllables

def load_lm_dict(filename):
    """Load an LM dictionary file from data/reference/."""
    path = os.path.join("data", "reference", filename)
    with open(path, encoding="utf-8") as f:
        return set(nltk.tokenize.word_tokenize(f.read().lower()))

def fog_index(tokens, sentences, numsyllables_fn):
    N_words = len(tokens)
    N_sents = len(sentences)
    doc_nonstopwords = [t for t in tokens if t not in stop_words]
    N_nonstopwords = len(doc_nonstopwords)
    complexwords = [w for w in doc_nonstopwords if numsyllables_fn(w) > 2]
    N_complex = len(complexwords)
    return 0.4 * ((N_words / N_sents) + (N_complex / N_nonstopwords)) if N_sents and N_nonstopwords else 0

def flesch_reading_ease(tokens, sentences, numsyllables_fn):
    nopunctuation = [t for t in tokens if t not in string.punctuation]
    total_words = len(nopunctuation)
    total_sentences = len(sentences)
    total_syllables = sum(numsyllables_fn(w) for w in tokens)
    return 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words) if total_sentences and total_words else 0

def analyze_filings(tickers, pronouncing_dict):
    """Run Uncertainty, Tone, FOG, and Flesch in one pass per filing."""
    numsyllables_fn = make_numsyllables(pronouncing_dict)

    # Load LM dictionaries once
    uncertainty_set = load_lm_dict("LM_Uncertainty.txt")
    positive_set    = load_lm_dict("LM_Positive.txt")
    negative_set    = load_lm_dict("LM_Negative.txt")

    results = {
        "Ticker": [],
        "Uncertainty": [],
        "Tone": [],
        "FOG": [],
        "Readability": []
    }

    for ticker in tickers:
        t0 = time.time()
        filing_path = os.path.join(SEC_FILINGS_PATH, f"{ticker}_10K.txt")
        if not os.path.exists(filing_path):
            print(f"⚠️ Missing 10-K for {ticker}")
            continue

        with open(filing_path, encoding="utf-8") as f:
            raw_text = f.read()
        print(f"{ticker} - Read file: {time.time() - t0:.2f}s")

        # Clean HTML tags
        t1 = time.time()
        clean_text = re.sub(r'<[^>]+>', ' ', raw_text.lower())
        print(f"{ticker} - HTML strip: {time.time() - t1:.2f}s")

        # Tokenize
        t2 = time.time()
        tokens = nltk.tokenize.word_tokenize(clean_text)
        sentences = nltk.tokenize.sent_tokenize(clean_text)
        print(f"{ticker} - Tokenization: {time.time() - t2:.2f}s")

        # Stopword removal + lemmatization
        t3 = time.time()
        nonstop = [t for t in tokens if t not in stop_words and t.isalpha()]
        lemmawords = [lemma.lemmatize(t) for t in nonstop]
        print(f"{ticker} - Stopword removal + lemmatization: {time.time() - t3:.2f}s")

        # --- Metric calculations ---
        # Uncertainty: proportion of lemmawords in uncertainty_set
        uncertainty_count = sum(1 for w in lemmawords if w in uncertainty_set)
        uncertainty_score = uncertainty_count / len(lemmawords) if lemmawords else 0

        # Tone: (positive - negative) / total sentiment words
        pos_count = sum(1 for w in lemmawords if w in positive_set)
        neg_count = sum(1 for w in lemmawords if w in negative_set)
        total_sentiment = pos_count + neg_count
        tone_score = ((pos_count - neg_count) / total_sentiment) if total_sentiment else 0

        # FOG index
        fog_score = fog_index(tokens, sentences, numsyllables_fn)

        # Flesch Reading Ease
        readability_score = flesch_reading_ease(tokens, sentences, numsyllables_fn)

        # Append to results
        results["Ticker"].append(ticker)
        results["Uncertainty"].append(uncertainty_score)
        results["Tone"].append(tone_score)
        results["FOG"].append(fog_score)
        results["Readability"].append(readability_score)

    return pd.DataFrame(results)
