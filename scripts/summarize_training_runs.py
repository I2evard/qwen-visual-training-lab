from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize local DirectML LoRA training run metrics."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root / "runs",
        help="Directory containing per-run metrics.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path for the summary.",
    )
    parser.add_argument(
        "--fail-on-latest-regression",
        action="store_true",
        help="Exit non-zero if the latest run has higher final eval loss than initial eval loss.",
    )
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def run_record(metrics_path: Path) -> dict[str, Any] | None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    initial_eval_loss = safe_float(metrics.get("initial_eval_loss"))
    final_eval_loss = safe_float(metrics.get("final_eval_loss"))
    if initial_eval_loss is None or final_eval_loss is None:
        return None

    config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    eval_loss_delta = safe_float(metrics.get("eval_loss_delta"))
    if eval_loss_delta is None:
        eval_loss_delta = final_eval_loss - initial_eval_loss

    improvement_pct = safe_float(metrics.get("eval_loss_improvement_pct"))
    if improvement_pct is None and initial_eval_loss:
        improvement_pct = ((initial_eval_loss - final_eval_loss) / initial_eval_loss) * 100

    return {
        "run_dir": str(metrics_path.parent),
        "metrics_path": str(metrics_path),
        "modified_utc": metrics_path.stat().st_mtime,
        "status": metrics.get("status"),
        "dataset_path": config.get("dataset_path"),
        "train_examples": metrics.get("train_examples"),
        "eval_examples": metrics.get("eval_examples"),
        "vocab_size": metrics.get("vocab_size"),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "eval_loss_delta": eval_loss_delta,
        "eval_loss_improvement_pct": improvement_pct,
        "final_train_loss": metrics.get("final_train_loss"),
        "mean_train_loss": metrics.get("mean_train_loss"),
        "epochs": config.get("epochs"),
        "max_seq_len": config.get("max_seq_len"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "lora_rank": config.get("lora_rank"),
        "learning_rate": config.get("learning_rate"),
    }


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        record = run_record(metrics_path)
        if record is not None:
            records.append(record)
    return records


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(records, key=lambda record: record["final_eval_loss"])
    latest = max(records, key=lambda record: record["modified_utc"])
    improved_count = sum(1 for record in records if record["eval_loss_delta"] < 0)
    return {
        "run_count": len(records),
        "improved_run_count": improved_count,
        "best_run": best,
        "latest_run": latest,
        "runs": sorted(records, key=lambda record: record["final_eval_loss"]),
    }


def print_summary(summary: dict[str, Any]) -> None:
    best = summary["best_run"]
    latest = summary["latest_run"]
    print(f"TRAINING_RUNS={summary['run_count']}")
    print(f"IMPROVED_RUNS={summary['improved_run_count']}")
    print(f"BEST_RUN={best['run_dir']}")
    print(f"BEST_FINAL_EVAL_LOSS={best['final_eval_loss']:.6f}")
    print(f"BEST_EVAL_LOSS_DELTA={best['eval_loss_delta']:.6f}")
    if best["eval_loss_improvement_pct"] is not None:
        print(f"BEST_EVAL_IMPROVEMENT_PCT={best['eval_loss_improvement_pct']:.2f}")
    print(f"LATEST_RUN={latest['run_dir']}")
    print(f"LATEST_FINAL_EVAL_LOSS={latest['final_eval_loss']:.6f}")
    print(f"LATEST_EVAL_LOSS_DELTA={latest['eval_loss_delta']:.6f}")
    if latest["eval_loss_improvement_pct"] is not None:
        print(f"LATEST_EVAL_IMPROVEMENT_PCT={latest['eval_loss_improvement_pct']:.2f}")


def main() -> int:
    args = parse_args()
    records = load_runs(args.runs_dir)
    if not records:
        print(f"NO_TRAINING_RUN_METRICS_FOUND={args.runs_dir}")
        return 1

    summary = build_summary(records)
    print_summary(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"SUMMARY_WRITTEN={args.output}")

    latest = summary["latest_run"]
    if args.fail_on_latest_regression and latest["eval_loss_delta"] > 0:
        print("LATEST_RUN_REGRESSED")
        return 2

    print("TRAINING_RUN_SUMMARY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
