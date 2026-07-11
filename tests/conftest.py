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

    def __init__(self, ok=True, hits=None, find_hits=None):
        self._ok = ok
        self._hits = list(hits or [])
        self._find_hits = list(find_hits or [])

    def ping(self):
        return self._ok

    def count(self):
        return len(self._hits) or 0

    def search(self, vector, limit=10, mood=None):
        if not self._ok:
            raise RuntimeError("qdrant down")
        out = [h for h in self._hits if mood is None or h.mood == mood]
        return out[:limit]

    def find_song(self, title, artist=None, limit=5):
        if not self._ok:
            raise RuntimeError("qdrant down")
        return self._find_hits[:limit]


def build_tiny_onnx(vocab_size: int, n_labels: int, out_path):
    """Hand-built ONNX graph with the real serving I/O contract:
    Gather(embedding, input_ids) -> ReduceMean(axis=1) -> logits.
    attention_mask is a declared (unused) input so the serving code's feed
    dict matches a real DistilBERT export."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(42)
    emb = rng.normal(scale=0.5, size=(vocab_size, n_labels)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Gather", ["emb", "input_ids"], ["tok_emb"]),
            helper.make_node("ReduceMean", ["tok_emb"], ["logits"], axes=[1], keepdims=0),
        ],
        name="tiny_mood",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", n_labels])],
        initializer=[numpy_helper.from_array(emb, name="emb")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


@pytest.fixture(scope="session")
def tiny_onnx_dir(tmp_path_factory):
    """models/transformer-shaped artifact dir: model.onnx + tokenizer.json + labels.json."""
    import json

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    d = tmp_path_factory.mktemp("tiny_transformer")

    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(
        [t for t, _ in TINY_SONGS], WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"])
    )
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    tok.enable_truncation(max_length=32)
    tok.save(str(d / "tokenizer.json"))

    labels = ["Angry", "Calm", "Hype", "Romantic", "Sad"]
    (d / "labels.json").write_text(json.dumps(labels))

    build_tiny_onnx(vocab_size=tok.get_vocab_size(), n_labels=len(labels), out_path=d / "model.onnx")
    return d


def build_tiny_embedder_onnx(vocab_size: int, dim: int, out_path):
    """Tiny embedding graph with the real embedder I/O contract:
    Gather(table, input_ids) -> token_embeddings [batch, seq, dim].
    attention_mask declared (unused) so the serving feed dict matches."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(42)
    table = rng.normal(scale=0.5, size=(vocab_size, dim)).astype(np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Gather", ["table", "input_ids"], ["token_embeddings"])],
        name="tiny_embedder",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        outputs=[
            helper.make_tensor_value_info("token_embeddings", TensorProto.FLOAT, ["batch", "seq", dim])
        ],
        initializer=[numpy_helper.from_array(table, name="table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


@pytest.fixture(scope="session")
def tiny_embedder_dir(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    d = tmp_path_factory.mktemp("tiny_embedder")
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(
        [t for t, _ in TINY_SONGS], WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"])
    )
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    tok.enable_truncation(max_length=32)
    tok.save(str(d / "tokenizer.json"))
    build_tiny_embedder_onnx(vocab_size=tok.get_vocab_size(), dim=8, out_path=d / "model.onnx")
    return d


class FakeEmbedder:
    """Deterministic within a process (hash-seeded)."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts):
        import numpy as np

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=self._dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out
