"""
Preprocessing helpers — text cleaning, mood labels, gap-zone filtering.

Mood labels come from Spotify's valence (positivity) and energy (intensity).
Based on Russell's circumplex idea: the 2D valence-energy space gets cut
into 5 mood regions + a central "gap zone" I drop because those songs
don't clearly belong to any mood.

AI attribution: implementation by Claude (Anthropic) based on my specification.
I chose the 5-mood taxonomy, the threshold values (0.3/0.6 valence, 0.4/0.6
energy), the gap-zone rule, and the clean_text behavior. Claude wrote the
function bodies. I reviewed and tested all output. See ../ATTRIBUTION.md
for the full breakdown.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# thresholds on valence and energy (both 0-1)
VALENCE_LOW = 0.3
VALENCE_HIGH = 0.6
ENERGY_LOW = 0.4
ENERGY_HIGH = 0.6

MOODS = ["Hype", "Romantic", "Calm", "Sad", "Angry"]

STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Lowercase, strip Genius section headers + punctuation, drop stopwords."""
    if not isinstance(text, str):
        return ""
    # strip stuff like [Chorus], [Verse 1: Artist]
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.lower()
    # keep letters and spaces only
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def derive_mood_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'mood' column based on valence/energy thresholds.

    Gap-zone rows (middle of both axes) get NaN — filter them out
    before training.
    """
    out = df.copy()
    v = out["valence"].astype(float)
    e = out["energy"].astype(float)

    mood = pd.Series(index=out.index, dtype="object")

    # high valence
    mood[(v >= VALENCE_HIGH) & (e >= ENERGY_HIGH)] = "Hype"
    mood[(v >= VALENCE_HIGH) & (e < ENERGY_HIGH)] = "Romantic"
    # low valence
    mood[(v < VALENCE_LOW) & (e >= ENERGY_HIGH)] = "Angry"
    mood[(v < VALENCE_LOW) & (e < ENERGY_HIGH)] = "Sad"
    # mid valence — Calm when low energy, Hype when high, gap in the middle
    mid_v = (v >= VALENCE_LOW) & (v < VALENCE_HIGH)
    mood[mid_v & (e < ENERGY_LOW)] = "Calm"
    mood[mid_v & (e >= ENERGY_HIGH)] = "Hype"

    out["mood"] = mood
    return out


def is_gap_zone(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True where a song falls in the ambiguous middle."""
    v = df["valence"].astype(float)
    e = df["energy"].astype(float)
    return (
        (v >= VALENCE_LOW) & (v < VALENCE_HIGH)
        & (e >= ENERGY_LOW) & (e < ENERGY_HIGH)
    )


def filter_gap_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows in the gap zone. Returns a new DataFrame."""
    return df[~is_gap_zone(df)].copy()


def get_class_weights(y: pd.Series) -> dict:
    """Balanced weights — inverse class frequency, normalized."""
    counts = y.value_counts()
    n = len(y)
    k = len(counts)
    return {cls: n / (k * cnt) for cls, cnt in counts.items()}
