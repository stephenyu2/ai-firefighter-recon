"""
Simple heuristic baselines for Tier 1 (fire/no-fire detection). No model,
no training - just a heuristic rule, to give run_eval.py a floor to compare
the fine-tuned detector against.

Two strategies:
  - "majority": predict whatever class is more common in the train split.
    For FLAME 3's ~5:1 fire:no-fire ratio, this means always predicting
    fire - same behavior as "always_fire" below, but derived from data
    rather than hardcoded, so it still makes sense if you re-run this on
    a differently-balanced split or dataset.
  - "always_fire": always predict fire, regardless of the train split.
    Kept as an explicit separate option because it's the clearest
    illustration of recall being gameable - perfect recall, poor
    precision - worth having as a fixed reference point independent of
    whatever the current class balance happens to be.

Usage:
    python simple_baseline.py --train data/splits/train.csv \
        --test data/splits/test.csv --strategy majority \
        --out results/predictions/simple_baseline_majority.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def fit_majority_class(train_df: pd.DataFrame, label_col: str = "label"):
    """Returns whichever label value is most frequent in the train split."""
    counts = train_df[label_col].value_counts()
    if counts.empty:
        raise ValueError("train split has no labeled rows - cannot fit majority baseline")
    return counts.idxmax()


def predict(test_df: pd.DataFrame, strategy: str, majority_label=None, fire_label=1):
    if strategy == "majority":
        if majority_label is None:
            raise ValueError("majority_label must be provided for strategy='majority'")
        return [majority_label] * len(test_df)
    elif strategy == "always_fire":
        return [fire_label] * len(test_df)
    elif strategy == "always_no_fire":
        no_fire_label = 0 if fire_label == 1 else fire_label
        return [no_fire_label] * len(test_df)
    else:
        raise ValueError(f"unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path, help="train.csv from make_splits.py")
    parser.add_argument("--test", required=True, type=Path, help="test.csv from make_splits.py")
    parser.add_argument("--strategy", choices=["majority", "always_fire", "always_no_fire"], default="majority")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--fire_label", type=int, default=1)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    majority_label = None
    if args.strategy == "majority":
        majority_label = fit_majority_class(train_df, args.label_col)
        majority_share = (train_df[args.label_col] == majority_label).mean()
        print(f"majority class in train split: {majority_label} "
              f"({majority_share:.1%} of {len(train_df)} rows)")

    preds = predict(test_df, args.strategy, majority_label=majority_label, fire_label=args.fire_label)

    out_df = test_df[["sample_id", args.label_col]].copy()
    out_df = out_df.rename(columns={args.label_col: "label"})
    out_df["prediction"] = preds
    out_df["strategy"] = args.strategy

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"wrote {len(out_df)} predictions to {args.out}")


if __name__ == "__main__":
    main()