"""
Fine-tune DistilBERT for 5-mood classification and export to quantized ONNX.

Designed for Colab free GPU (full mode, ~1-2h) but fully testable locally:
--smoke runs the identical pipeline with a tiny random-init config on CPU in
minutes. Artifacts land in --out as model.onnx + tokenizer.json + labels.json
(+ metrics.json), the exact contract api/services/transformer.py loads.

Split discipline: identical to the notebooks — src.classify.split_data,
random_state=42, stratified. The test split is NEVER touched here; final test
metrics come from training/evaluate.py only.

Usage (Colab):   python training/finetune_distilbert.py --data data/processed/songs_labeled.csv --out models/transformer
Usage (smoke):   python training/finetune_distilbert.py --data data/processed/songs_labeled.csv --out /tmp/smoke_transformer --smoke

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.2 training recipe: max_len 256, 2-3 epochs, class-weighted
loss, early stopping on val macro F1, int8 dynamic quantization).
See ../ATTRIBUTION.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_LEN = 256
MODEL_NAME = "distilbert-base-uncased"


def load_splits(data_path: str):
    from src.classify import split_data

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["lyrics", "mood"]).reset_index(drop=True)
    idx = df.index.to_numpy()
    X_tr, X_val, _, _, _, _ = split_data(idx, df["mood"], random_state=42)
    return df.loc[X_tr], df.loc[X_val]


def class_weights(y: pd.Series, labels: list[str]) -> np.ndarray:
    counts = y.value_counts()
    n, k = len(y), len(labels)
    return np.array([n / (k * counts[l]) for l in labels], dtype=np.float32)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny model, tiny data, CPU, minutes")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, DistilBertConfig, DistilBertForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_splits(args.data)
    if args.smoke:
        train_df, val_df = train_df.iloc[:48], val_df.iloc[:16]
        args.epochs, args.batch_size = 1, 8

    labels = sorted(train_df["mood"].unique())
    label_to_id = {l: i for i, l in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.smoke:
        config = DistilBertConfig(
            vocab_size=tokenizer.vocab_size, n_layers=2, dim=64, n_heads=2,
            hidden_dim=128, num_labels=len(labels), max_position_embeddings=512,
        )
        model = DistilBertForSequenceClassification(config)
    else:
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels))
    model.to(device)

    def encode(df: pd.DataFrame) -> TensorDataset:
        enc = tokenizer(
            df["lyrics"].astype(str).tolist(), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        y = torch.tensor([label_to_id[m] for m in df["mood"]], dtype=torch.long)
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    train_dl = DataLoader(encode(train_df), batch_size=args.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
    val_dl = DataLoader(encode(val_df), batch_size=args.batch_size)

    weights = torch.tensor(class_weights(train_df["mood"], labels)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def val_macro_f1() -> float:
        from sklearn.metrics import f1_score

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, mask, y in val_dl:
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                preds += logits.argmax(-1).cpu().tolist()
                trues += y.tolist()
        return float(f1_score(trues, preds, average="macro"))

    best_f1, best_state, epochs_run = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for step, (ids, mask, y) in enumerate(train_dl):
            optim.zero_grad()
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            optim.step()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        f1 = val_macro_f1()
        epochs_run = epoch + 1
        print(f"epoch {epoch}: val macro F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print("early stop: val macro F1 did not improve")
            break

    model.load_state_dict(best_state)
    model.cpu().eval()

    # --- export: ONNX with the serving I/O contract, then int8 dynamic quantization
    dummy = tokenizer(["dummy lyrics for export"], return_tensors="pt", padding=True)
    model.config.return_dict = False  # tuple outputs so torch.onnx.export maps output_names cleanly
    fp32_path = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # torch>=2.9 defaults to the dynamo exporter (needs onnxscript);
                       # force the legacy TorchScript exporter these dynamic_axes/opset args target
    )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(out_dir / "model.onnx"), weight_type=QuantType.QInt8)
    fp32_path.unlink()

    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "labels.json").write_text(json.dumps(labels))
    (out_dir / "metrics.json").write_text(json.dumps(
        {"best_val_macro_f1": best_f1, "epochs_run": epochs_run, "smoke": args.smoke}
    ))

    print(f"artifacts written to {out_dir} (best val macro F1: {best_f1:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
