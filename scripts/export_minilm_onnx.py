"""
One-time export: all-MiniLM-L6-v2 -> models/embedder/{model.onnx,tokenizer.json}.

fp32 (NOT quantized) — query vectors must match the corpus embedding space,
so we verify parity against sentence-transformers before declaring success.
Run locally (torch required): python scripts/export_minilm_onnx.py

AI attribution: implementation by Claude (Anthropic) based on my specification.
See ../ATTRIBUTION.md.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PARITY_SENTENCES = [
    "stadium lights and a roaring crowd",
    "rain on the empty street tonight",
    "your hand in mine by the kitchen door",
    "tea gone cold by the window",
    "say it again, say it to my face",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/embedder")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.config.return_dict = False

    # transformers 5.x decorates forward() so the legacy TorchScript tracer
    # mis-binds positional args; wrap with an explicit keyword-arg forward that
    # returns only last_hidden_state ([batch, seq, dim] == token_embeddings).
    class _ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    export_model = _ExportWrapper(model)

    dummy = tokenizer(["dummy text for export"], return_tensors="pt", padding=True)
    torch.onnx.export(
        export_model, (dummy["input_ids"], dummy["attention_mask"]), str(out_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["token_embeddings"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "token_embeddings": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))

    # parity check vs sentence-transformers
    from sentence_transformers import SentenceTransformer

    from api.services.embedder import load_embedder

    ours = load_embedder(out_dir).embed(PARITY_SENTENCES)
    ref = SentenceTransformer("all-MiniLM-L6-v2").encode(
        PARITY_SENTENCES, normalize_embeddings=True, convert_to_numpy=True
    )
    cos = (ours * ref).sum(axis=1)
    worst = float(1.0 - cos.min())
    print(f"parity: worst cosine deviation = {worst:.2e}")
    if worst > 1e-3:
        print("PARITY FAIL — do not serve this export")
        return 1
    print(f"PASS — artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
