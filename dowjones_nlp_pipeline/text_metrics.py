import os
import pandas as pd
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import SEC_FILINGS_PATH

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def load_lm_dict(filename):
    """Load an LM dictionary file from data/reference/."""
    path = os.path.join("data", "reference", filename)
    with open(path, encoding="utf-8") as f:
        return set(nltk.tokenize.word_tokenize(f.read().lower()))

def numsyllables(word, pronouncing_dict):
    try:
        syll_list = [x for x in pronouncing_dict[word][0] if x[-1].isdigit()]
        return len(syll_list)
    except:
        return 0

def fog_index(text, pronouncing_dict):
    docwords = nltk.tokenize.word_tokenize(text.lower())
    docsents = nltk.tokenize.sent_tokenize(text.lower())
    N_words = len(docwords)
    N_sents = len(docsents)
    doc_nonstopwords = [t for t in docwords if t not in stop_words]
    N_nonstopwords = len(doc_nonstopwords)
    complexwords = [w for w in doc_nonstopwords if numsyllables(w, pronouncing_dict) > 2]
    N_complex = len(complexwords)
    return 0.4 * ((N_words / N_sents) + (N_complex / N_nonstopwords))

def flesch_reading_ease(text, pronouncing_dict):
    sampletokens = nltk.tokenize.word_tokenize(text.lower())
    total_sentences = len(nltk.tokenize.sent_tokenize(text))
    nopunctuation = [t for t in sampletokens if t not in string.punctuation]
    total_words = len(nopunctuation)
    total_syllables = sum(numsyllables(w, pronouncing_dict) for w in sampletokens)
    return 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)

def analyze_filings(tickers, pronouncing_dict):
    """Run Uncertainty, Tone, FOG, and Flesch in one pass per filing."""
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
        filing_path = os.path.join(SEC_FILINGS_PATH, f"{ticker}_10K.txt")
        if not os.path.exists(filing_path):
            print(f"⚠️ Missing 10-K for {ticker}")
            continue

        with open(filing_path, encoding="utf-8") as f:
            raw_text = f.read()

        # Clean HTML
        clean_text = BeautifulSoup(raw_text.lower(), "html.parser").get_text()

        # Tokenize, remove stopwords, lemmatize
        tokens = nltk.tokenize.word_tokenize(clean_text)
        nonstop = [t for t in tokens if t not in stop_words]
        lemmawords = [lemma.lemmatize(t) for t in nonstop]

        # Uncertainty %
        pct_uncertainty = (len([w for w in lemmawords if w in uncertainty_set]) / len(lemmawords)) if lemmawords else 0

        # Tone = Positive % − Negative %
        pct_pos = (len([w for w in lemmawords if w in positive_set]) / len(lemmawords)) if lemmawords else 0
        pct_neg = (len([w for w in lemmawords if w in negative_set]) / len(lemmawords)) if lemmawords else 0
        tone_score = pct_pos - pct_neg

        # Readability metrics
        fog_score = fog_index(clean_text, pronouncing_dict)
        flesch_score = flesch_reading_ease(clean_text, pronouncing_dict)

        # Store results
        results["Ticker"].append(ticker)
        results["Uncertainty"].append(pct_uncertainty)
        results["Tone"].append(tone_score)
        results["FOG"].append(fog_score)
        results["Readability"].append(flesch_score)

    return pd.DataFrame(results)
