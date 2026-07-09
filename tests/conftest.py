"""
Shared fixtures: a real (tiny) sklearn model for service tests, and a
FakeMoodModel + FakeRetrieval for API route tests so no heavy artifacts
are needed in CI.
"""

import pytest

# 3 songs per mood, words chosen to survive clean_text stopword stripping
TINY_SONGS = [
    ("stadium lights bass kicking loud crowd jumping", "Hype"),
    ("party anthem hands raised speakers booming dance", "Hype"),
    ("energy rising drums pounding neon strobing night", "Hype"),
    ("tender heart kitchen door soft radio lifetime", "Romantic"),
    ("gentle kisses warm embrace candle dinner roses", "Romantic"),
    ("holding hands moonlight promise sweet whisper darling", "Romantic"),
    ("slow light rug cool tea window tree quiet", "Calm"),
    ("morning stillness breeze garden peaceful drifting cloud", "Calm"),
    ("lazy sunday blanket humming kettle soft rain", "Calm"),
    ("rain empty street counted cars coat chair alone", "Sad"),
    ("tears falling goodbye letter fading photograph missing", "Sad"),
    ("grey sky lonely echo hollow rooms winter grief", "Sad"),
    ("face blame burned door shouting fists slammed", "Angry"),
    ("rage boiling betrayal lies screaming broken glass", "Angry"),
    ("fury storm smashed walls venom spite revenge", "Angry"),
]


@pytest.fixture(scope="session")
def tiny_model():
    """A real BaselineMoodModel trained on TINY_SONGS — fast, deterministic."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from api.services.model import BaselineMoodModel

    texts = [t for t, _ in TINY_SONGS]
    labels = [m for _, m in TINY_SONGS]
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, labels)
    return BaselineMoodModel(clf=clf, vectorizer=vec, background=None, version="test-v0")


class FakeMoodModel:
    """Canned-response model for route tests."""

    version = "fake-v0"

    def __init__(self, mood="Hype", confidence=0.9, explanation=(("stadium", 0.5),), fail=False):
        self._mood = mood
        self._confidence = confidence
        self._explanation = list(explanation)
        self._fail = fail

    def predict(self, lyrics, explain=True):
        from api.services.model import PredictionResult

        if self._fail:
            raise RuntimeError("model exploded")
        return PredictionResult(
            mood=self._mood,
            confidence=self._confidence,
            probabilities={self._mood: self._confidence},
            explanation=self._explanation if explain else None,
        )


class FakeRetrieval:
    """Canned retrieval client for route tests."""

    def __init__(self, ok=True):
        self._ok = ok

    def ping(self):
        return self._ok
