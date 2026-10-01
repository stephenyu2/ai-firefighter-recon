"""
Tier 2 evaluation harness - direction/trend agreement, scored separately
from Tier 1 since it depends on the small human-labeled evaluation sample
(~50-100 sequences) rather than the automated FLAME 3 / Boreal test split.

Each predictions CSV here needs: sequence_id, label (the human-assigned
ground truth - a trend class like "growing"/"stable"/"receding", or a
direction bin), prediction (the method's output in the same label space).
Build one CSV per method (deterministic mask-tracking baseline, Molmo 2)
the same way simple_baseline.py / reference_baseline.py produce Tier 1
predictions CSVs.

This script does two things:
  1. Scores each method against the human-labeled ground truth
     (accuracy + Cohen's kappa, via metrics.compute_trend_metrics).
  2. Optionally computes agreement BETWEEN two methods directly (e.g.
     mask-tracking vs. Molmo 2) - where they disagree with each other is
     often more informative than either accuracy number alone, per the
     evaluation design we settled on earlier.

Usage:
    python eval/run_tier2_eval.py \
        --predictions results/predictions/tier2_mask_tracking.csv \
                       results/predictions/tier2_molmo2.csv \
        --agree_pair results/predictions/tier2_mask_tracking.csv \
                      results/predictions/tier2_molmo2.csv \
        --out results/tier2_results
"""
import argparse
from pathlib import Path

import pandas as pd

from metrics import compute_trend_metrics, format_metrics_table


def score_against_human_labels(path: Path, label_col: str = "label", pred_col: str = "prediction", name: str = None):
    df = pd.read_csv(path)
    for col in (label_col, pred_col):
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column '{col}' - got columns {list(df.columns)}")

    metrics = compute_trend_metrics(df[label_col], df[pred_col])
    row = {"name": name or path.stem}
    row.update(metrics.as_dict())
    return row


def score_inter_method_agreement(path_a: Path, path_b: Path,
                                  id_col: str = "sequence_id", pred_col: str = "prediction"):
    """
    Agreement between two methods' predictions on the SAME sequences -
    not against human ground truth. Joins on id_col so this still works
    if the two CSVs aren't in the same row order.
    """
    df_a = pd.read_csv(path_a)[[id_col, pred_col]].rename(columns={pred_col: "pred_a"})
    df_b = pd.read_csv(path_b)[[id_col, pred_col]].rename(columns={pred_col: "pred_b"})
    merged = df_a.merge(df_b, on=id_col, how="inner")

    if len(merged) < len(df_a) or len(merged) < len(df_b):
        dropped = (len(df_a) + len(df_b)) - 2 * len(merged)
        print(f"[warn] {dropped} sequence(s) didn't match between {path_a.name} and {path_b.name} - "
              f"scoring only the {len(merged)} overlapping sequences")

    metrics = compute_trend_metrics(merged["pred_a"], merged["pred_b"])
    row = {"name": f"{path_a.stem} vs {path_b.stem}"}
    row.update(metrics.as_dict())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", default=[],
                         help="predictions CSVs to score against human-labeled ground truth")
    parser.add_argument("--agree_pair", type=Path, nargs=2, default=None,
                         help="two predictions CSVs to score against EACH OTHER instead of ground truth")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--pred_col", default="prediction")
    parser.add_argument("--id_col", default="sequence_id")
    parser.add_argument("--out", type=Path, default=None,
                         help="path prefix to save results (writes <prefix>.csv and <prefix>.md)")
    args = parser.parse_args()

    rows = []
    for path in args.predictions:
        if not path.exists():
            print(f"[warn] skipping missing file: {path}")
            continue
        rows.append(score_against_human_labels(path, args.label_col, args.pred_col))

    if args.agree_pair:
        path_a, path_b = args.agree_pair
        if path_a.exists() and path_b.exists():
            rows.append(score_inter_method_agreement(path_a, path_b, args.id_col, args.pred_col))
        else:
            print(f"[warn] skipping agree_pair - one or both files missing: {path_a}, {path_b}")

    if not rows:
        print("[error] nothing was scored - check --predictions / --agree_pair paths")
        return

    table_text = format_metrics_table(rows)
    print(table_text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(rows)
        results_df.to_csv(args.out.with_suffix(".csv"), index=False)

        md_lines = ["| " + " | ".join(results_df.columns) + " |",
                    "|" + "|".join(["---"] * len(results_df.columns)) + "|"]
        for _, r in results_df.iterrows():
            md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
        args.out.with_suffix(".md").write_text("\n".join(md_lines))

        print(f"\nwrote results to {args.out.with_suffix('.csv')} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()