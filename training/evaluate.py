"""
Eval harness — one command to score any registered model on the frozen test
split, write a markdown report, and (optionally) log the run to MLflow.

The split is byte-identical to the notebooks' split: same
src.classify.split_data, same random_state=42, stratified on mood.

Usage:
    python -m training.evaluate --model baseline
    python -m training.evaluate --model transformer --limit 500 --no-mlflow

AI attribution: implementation by Claude (Anthropic) based on my specification
(design spec §3.2 eval harness + quality gate). See ../ATTRIBUTION.md.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classify import split_data  # noqa: E402


def frozen_test_split(df: pd.DataFrame) -> pd.DataFrame:
    """The exact test rows the notebooks used (random_state=42, stratified)."""
    idx = df.index.to_numpy()
    _, _, X_test, _, _, _ = split_data(idx, df["mood"], random_state=42)
    return df.loc[X_test]


def evaluate_predictor(model, df: pd.DataFrame, text_col: str = "lyrics",
                       label_col: str = "mood", limit: int | None = None) -> dict:
    rows = df if limit is None else df.iloc[:limit]
    y_true = rows[label_col].tolist()
    y_pred = [model.predict(t, explain=False).mood for t in rows[text_col]]

    classes = sorted(set(y_true) | set(y_pred))
    p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, pr in zip(y_true, y_pred):
        confusion[t][pr] += 1

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_precision": {c: float(v) for c, v in zip(classes, p)},
        "per_class_recall": {c: float(v) for c, v in zip(classes, r)},
        "confusion": {t: dict(d) for t, d in confusion.items()},
        "n": len(rows),
        "model_version": model.version,
    }


def majority_baseline_macro_f1(y) -> float:
    majority = Counter(y).most_common(1)[0][0]
    return float(f1_score(list(y), [majority] * len(y), average="macro"))


def passes_quality_gate(results: dict, y) -> bool:
    return results["macro_f1"] > majority_baseline_macro_f1(y)


def write_report(results: dict, name: str, out_dir: Path = Path("results")) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(results["per_class_precision"])

    lines = [
        f"# Eval report — {name} ({results['model_version']})",
        "",
        f"- n: {results['n']}",
        f"- accuracy: {results['accuracy']:.4f}",
        f"- macro_f1: {results['macro_f1']:.4f}",
        "",
        "| class | precision | recall |",
        "|---|---|---|",
    ]
    lines += [
        f"| {c} | {results['per_class_precision'][c]:.3f} | {results['per_class_recall'][c]:.3f} |"
        for c in classes
    ]
    lines += ["", "## Confusion (true → predicted)", "", "| true \\ pred | " + " | ".join(classes) + " |",
              "|---" * (len(classes) + 1) + "|"]
    for t in classes:
        row = results["confusion"].get(t, {})
        lines.append(f"| {t} | " + " | ".join(str(row.get(c, 0)) for c in classes) + " |")

    path = out_dir / f"eval_{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def log_mlflow(results: dict, name: str, params: dict) -> None:
    """Log to ./mlruns (file backend). Silently skips if mlflow isn't installed."""
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed — skipping tracking")
        return
    mlflow.set_experiment("lyricmood")
    with mlflow.start_run(run_name=f"eval-{name}"):
        mlflow.log_params({**params, "model_version": results["model_version"]})
        mlflow.log_metrics(
            {"accuracy": results["accuracy"], "macro_f1": results["macro_f1"], "n": results["n"]}
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a registered model on the frozen test split")
    parser.add_argument("--model", required=True, choices=["baseline", "transformer"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    from api.config import Settings
    from api.services.model import load_baseline
    from api.services.registry import load_registry
    from api.services.transformer import load_transformer

    settings = Settings()
    reg = load_registry(settings.registry_path)
    spec = reg.models[args.model]
    model = load_baseline(settings, version=spec.version) if spec.kind == "baseline" else load_transformer(spec.dir, spec.version)

    df = pd.read_csv(settings.labeled_songs_path)
    test_df = frozen_test_split(df)
    results = evaluate_predictor(model, test_df, limit=args.limit)

    print(f"model={args.model} version={results['model_version']} n={results['n']}")
    print(f"accuracy={results['accuracy']:.4f} macro_f1={results['macro_f1']:.4f}")
    report = write_report(results, args.model)
    print(f"report: {report}")

    if not args.no_mlflow:
        log_mlflow(results, args.model, params={"model": args.model, "limit": args.limit or 0})

    if not passes_quality_gate(results, test_df["mood"].iloc[: args.limit or len(test_df)]):
        print("QUALITY GATE FAILED: model does not beat majority-class macro F1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
