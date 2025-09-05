# utils/nlp_setup.py
from nltk.corpus import cmudict, stopwords
from nltk.stem import WordNetLemmatizer

# Load CMU Pronouncing Dictionary
d = cmudict.dict()

# Load stopwords
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemma = WordNetLemmatizer()

