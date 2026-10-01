"""
Text preprocessing utilities for SMS Spam Classification.

This module provides clean, function-based text normalization pipelines tailored
for classical NLP and SMS message analysis without aggressive stopword removal.
"""

import re
import string


def normalize_urls(text: str) -> str:
    """Replace URLs and website links with a standardized token."""
    # Match standard http/https links and www prefixes
    url_pattern = r"(https?://\S+|www\.\S+)"
    return re.sub(url_pattern, " urladdr ", text, flags=re.IGNORECASE)


def normalize_email_addresses(text: str) -> str:
    """Replace email addresses with a standardized token."""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    return re.sub(email_pattern, " emailaddr ", text)


def normalize_currencies(text: str) -> str:
    """Replace common currency symbols with a standardized token."""
    currency_pattern = r"[£$€₹¥]"
    return re.sub(currency_pattern, " currsym ", text)


def normalize_numbers(text: str) -> str:
    """
    Replace sequences of digits (phone numbers, shortcodes, pin codes)
    with a standardized number token while retaining word context.
    """
    return re.sub(r"\b\d+\b", " numtoken ", text)


def clean_text(text: str) -> str:
    """
    Clean and preprocess an SMS message for classical NLP vectorization.

    Steps:
    1. Convert input to string and handle empty/null cases.
    2. Lowercase the entire text.
    3. Normalize URLs to 'urladdr'.
    4. Normalize emails to 'emailaddr'.
    5. Normalize currency symbols to 'currsym'.
    6. Normalize standalone numbers to 'numtoken'.
    7. Remove punctuation (while preserving word boundaries).
    8. Collapse multi-space gaps into single whitespace and strip edges.

    Note: We purposefully DO NOT remove standard stopwords because key spam
    indicators (e.g., 'free', 'call', 'win', 'you', 'now', 'to') are critical
    for distinguishing spam from legitimate conversation.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Normalize URLs
    text = normalize_urls(text)

    # 3. Normalize Emails
    text = normalize_email_addresses(text)

    # 4. Normalize Currencies
    text = normalize_currencies(text)

    # 5. Normalize standalone numeric tokens
    text = normalize_numbers(text)

    # 6. Remove punctuation
    # Replace punctuation characters with whitespace to avoid gluing adjacent words
    trans_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(trans_table)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
