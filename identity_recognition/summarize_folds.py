import argparse
import json
import statistics
from pathlib import Path


DEFAULT_METRICS = (
    "roc_auc",
    "eer",
    "tar_at_far_0.1",
    "tar_at_far_0.01",
    "tar_at_far_0.001",
    "acc_sample",
    "acc_top5",
    "acc_subject",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize identity metrics across folds.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def summarize(records):
    summary = {"num_folds": len(records), "metrics": {}}
    for key in DEFAULT_METRICS:
        values = [float(record[key]) for record in records if key in record]
        if len(values) != len(records):
            continue
        summary["metrics"][key] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    return summary


def main():
    args = parse_args()
    records = [json.loads(path.read_text()) for path in args.inputs]
    summary = summarize(records)
    summary["inputs"] = [str(path) for path in args.inputs]
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"Fold summary saved to {args.out}")
    print(rendered)


if __name__ == "__main__":
    main()
