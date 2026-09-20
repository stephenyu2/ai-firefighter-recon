"""
Turn a FLAME 3 manifest into leakage-safe train/val/test splits, grouped by
time-block so near-duplicate consecutive frames never span two splits.

Boreal is NOT split here - by design it's reserved entirely as an
out-of-distribution generalization test set for whatever model you train
on FLAME 3's train split. If you later want a small held-out Boreal slice
for e.g. threshold calibration, split that off explicitly and document why.

Usage:
    python make_splits.py --manifest data/manifests/flame3.csv --out_dir data/splits --seed 42
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(df: pd.DataFrame, group_col: str, test_size: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, holdout_idx = next(gss.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[holdout_idx].reset_index(drop=True)


def report_balance(name: str, df: pd.DataFrame):
    counts = df["label"].value_counts(normalize=True).sort_index()
    print(f"  {name}: n={len(df)}, groups={df['group'].nunique()}, "
          f"class balance={counts.to_dict()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    # first peel off test, then split remainder into train/val
    train_val, test = grouped_split(df, "group", test_size=args.test_size, seed=args.seed)
    relative_val_size = args.val_size / (1 - args.test_size)
    train, val = grouped_split(train_val, "group", test_size=relative_val_size, seed=args.seed)

    # sanity check: no group should ever appear in more than one split
    train_groups, val_groups, test_groups = set(train["group"]), set(val["group"]), set(test["group"])
    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    assert not overlap, f"leakage detected - groups appear in multiple splits: {overlap}"

    print("Split sizes and class balance (verify these look reasonable before training):")
    report_balance("train", train)
    report_balance("val", val)
    report_balance("test", test)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.out_dir / "train.csv", index=False)
    val.to_csv(args.out_dir / "val.csv", index=False)
    test.to_csv(args.out_dir / "test.csv", index=False)
    print(f"wrote train/val/test CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()