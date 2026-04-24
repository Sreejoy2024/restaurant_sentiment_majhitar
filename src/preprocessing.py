"""
Text Preprocessing Module for Restaurant Review Sentiment Analysis
Majhitar, Sikkim - NLP Assignment
"""

import re
import string
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Optional


# ─────────────────────────────────────
# STOPWORDS (custom + NLTK fallback)
# ─────────────────────────────────────
CUSTOM_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who",
    "whom", "this", "that", "these", "those", "am", "to", "of", "and",
    "in", "that", "have", "it", "for", "not", "on", "with", "as", "at",
    "by", "from", "up", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "out", "off", "over", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "s", "t", "just", "don", "so", "no", "if", "or",
    "but", "also", "however", "although", "though", "while", "because",
    "since", "unless", "until", "whether", "yet", "still", "again",
}

# Food-domain positive/negative seed words (for feature engineering)
FOOD_POSITIVE_WORDS = {
    "delicious", "amazing", "tasty", "excellent", "wonderful", "fantastic",
    "great", "outstanding", "superb", "yummy", "fresh", "flavorful",
    "perfect", "loved", "best", "recommend", "enjoyable", "satisfied",
    "hot", "crispy", "authentic", "rich", "tender", "juicy",
}

FOOD_NEGATIVE_WORDS = {
    "terrible", "awful", "disgusting", "horrible", "bad", "worst",
    "tasteless", "bland", "cold", "stale", "undercooked", "overcooked",
    "overpriced", "slow", "dirty", "rude", "disappointing", "mediocre",
    "waste", "soggy", "oily", "salty", "bitter",
}


def clean_text(text: str) -> str:
    """Full text cleaning pipeline."""
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Normalize unicode (handles emojis, special chars)
    text = unicodedata.normalize("NFKD", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Expand common contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "i'm": "i am", "i've": "i have", "i'll": "i will",
        "i'd": "i would", "it's": "it is", "that's": "that is",
        "there's": "there is", "they're": "they are", "they've": "they have",
        "we're": "we are", "we've": "we have", "you're": "you are",
        "you've": "you have", "she's": "she is", "he's": "he is",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str, stopwords: Optional[set] = None) -> str:
    """Remove stopwords from cleaned text."""
    if stopwords is None:
        stopwords = CUSTOM_STOPWORDS
    tokens = text.split()
    filtered = [t for t in tokens if t not in stopwords and len(t) > 2]
    return " ".join(filtered)


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return text.split()


def get_word_count(text: str) -> int:
    return len(simple_tokenize(text))


def extract_features(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Feature engineering for sentiment analysis.
    Adds multiple text-derived features to the dataframe.
    """
    df = df.copy()

    # Basic cleaning
    df["cleaned_text"] = df[text_col].apply(clean_text)
    df["filtered_text"] = df["cleaned_text"].apply(remove_stopwords)

    # Length features
    df["review_length"] = df[text_col].apply(lambda x: len(str(x)))
    df["word_count"] = df["cleaned_text"].apply(get_word_count)
    df["char_count"] = df["cleaned_text"].apply(len)
    df["avg_word_length"] = df["cleaned_text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )

    # Punctuation features
    df["exclamation_count"] = df[text_col].apply(lambda x: str(x).count("!"))
    df["question_count"] = df[text_col].apply(lambda x: str(x).count("?"))

    # Sentiment indicator words
    df["positive_word_count"] = df["cleaned_text"].apply(
        lambda x: sum(1 for w in x.split() if w in FOOD_POSITIVE_WORDS)
    )
    df["negative_word_count"] = df["cleaned_text"].apply(
        lambda x: sum(1 for w in x.split() if w in FOOD_NEGATIVE_WORDS)
    )
    df["sentiment_word_ratio"] = (
        (df["positive_word_count"] - df["negative_word_count"]) /
        (df["word_count"].replace(0, 1))
    )

    # Negation check
    df["has_negation"] = df["cleaned_text"].apply(
        lambda x: int(any(w in x.split() for w in ["not", "never", "no", "neither", "nor"]))
    )

    print(f"[✓] Feature extraction complete. Added {len(df.columns) - 3} new features.")
    return df


def preprocess_pipeline(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Full preprocessing pipeline."""
    print(f"[→] Starting preprocessing pipeline on {len(df)} records...")

    # Drop nulls
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str)

    # Remove duplicate reviews
    original_len = len(df)
    df = df.drop_duplicates(subset=[text_col])
    print(f"[✓] Removed {original_len - len(df)} duplicate reviews.")

    # Feature extraction
    df = extract_features(df, text_col)

    print(f"[✓] Preprocessing complete. Final dataset: {len(df)} records.")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    # Test with a few samples
    sample_texts = [
        "The food was absolutely delicious! Best momos I've ever had.",
        "Terrible service. The chicken was cold and bland. Never coming back.",
        "Okay place, nothing special. The rice was decent.",
        "Loved the dal makhani here. Will definitely visit again!",
    ]

    df_test = pd.DataFrame({"review_text": sample_texts, "rating": [5, 1, 3, 4]})
    df_processed = preprocess_pipeline(df_test)
    print("\nSample processed output:")
    print(df_processed[["review_text", "cleaned_text", "word_count",
                          "positive_word_count", "negative_word_count"]].to_string())
