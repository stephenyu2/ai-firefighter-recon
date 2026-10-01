"""
Tier 1 evaluation harness. Scores one or more predictions CSVs (each with
columns: sample_id, label, prediction - the format simple_baseline.py and,
later, the fine-tuned model's inference script both produce) and prints /
saves a comparison table.

This is deliberately model-agnostic: it doesn't care whether "prediction"
came from a heuristic baseline, an off-the-shelf reference model, or your
own fine-tuned detector - it just needs a CSV in the right shape. That
means the fine-tuned model's inference script just needs to write
predictions in this format and it slots into the same harness with zero
changes here.

Example:
    python eval/run_eval.py \
        --predictions results/predictions/simple_baseline_majority.csv \
                       results/predictions/reference_baseline.csv \
        --out results/tier1_results

Example output (also printed to console):
    name                      | n  | recall | precision | f1     | accuracy | ...
    --------------------------+----+--------+-----------+--------+----------+----
    simple_baseline_majority  | 30 | 1.0000 | 0.8333    | 0.9091 | 0.8333   | ...
    reference_baseline        | 30 | 0.8800 | 0.9130    | 0.8961 | 0.9000   | ...
"""
import argparse
from pathlib import Path

import pandas as pd

from metrics import compute_detection_metrics, format_metrics_table


def score_predictions_file(path: Path, label_col: str = "label", pred_col: str = "prediction", name: str = None):
    df = pd.read_csv(path)
    for col in (label_col, pred_col):
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column '{col}' - got columns {list(df.columns)}")

    metrics = compute_detection_metrics(df[label_col], df[pred_col])
    row = {"name": name or path.stem}
    row.update(metrics.as_dict())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path, nargs="+",
                         help="one or more predictions CSVs to score and compare")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--pred_col", default="prediction")
    parser.add_argument("--out", type=Path, default=None,
                         help="path prefix to save results (writes <prefix>.csv and <prefix>.md)")
    args = parser.parse_args()

    rows = []
    for path in args.predictions:
        if not path.exists():
            print(f"[warn] skipping missing file: {path}")
            continue
        rows.append(score_predictions_file(path, args.label_col, args.pred_col))

    if not rows:
        print("[error] no predictions files were scored - nothing to report")
        return

    table_text = format_metrics_table(rows)
    print(table_text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(rows)
        csv_path = args.out.with_suffix(".csv")
        results_df.to_csv(csv_path, index=False)

        md_path = args.out.with_suffix(".md")
        md_lines = ["| " + " | ".join(results_df.columns) + " |",
                    "|" + "|".join(["---"] * len(results_df.columns)) + "|"]
        for _, r in results_df.iterrows():
            md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
        md_path.write_text("\n".join(md_lines))

        print(f"\nwrote results to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()