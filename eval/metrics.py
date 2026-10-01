"""
Shared metric functions for both evaluation scripts.

Tier 1 (fire/smoke detection): recall, precision, F1, accuracy, confusion
matrix. Recall is treated as primary throughout given the missed-fire vs.
false-alarm asymmetry.

Tier 2 (direction/trend agreement): accuracy and Cohen's kappa against a
small human-labeled sample. Kappa matters here specifically because trend
labels are imbalanced (most sequences trend "growing" during a burn), so
raw accuracy alone can look good from majority-class guessing.
"""
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
)


@dataclass
class DetectionMetrics:
    n: int
    recall: float
    precision: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    def as_dict(self):
        return asdict(self)


@dataclass
class TrendMetrics:
    n: int
    accuracy: float
    cohen_kappa: float

    def as_dict(self):
        return asdict(self)


def compute_detection_metrics(y_true, y_pred, positive_label=1) -> DetectionMetrics:
    """
    y_true / y_pred: array-like of 0/1 (or matching label values) for
    fire-present / fire-absent per sample. Order must correspond 1:1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred), "y_true and y_pred must be the same length"
    assert len(y_true) > 0, "cannot compute metrics on an empty set"

    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # confusion_matrix labels order: [negative, positive] assuming binary 0/1
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if len(labels) == 2 and positive_label in labels:
        pos_idx = labels.index(positive_label)
        neg_idx = 1 - pos_idx
        tp = int(cm[pos_idx, pos_idx])
        fn = int(cm[pos_idx, neg_idx])
        fp = int(cm[neg_idx, pos_idx])
        tn = int(cm[neg_idx, neg_idx])
    else:
        # degenerate case (e.g. only one class present) - fall back to zeros
        tp = fp = tn = fn = 0

    return DetectionMetrics(
        n=len(y_true),
        recall=round(float(recall), 4),
        precision=round(float(precision), 4),
        f1=round(float(f1), 4),
        accuracy=round(float(accuracy), 4),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def compute_trend_metrics(y_true, y_pred) -> TrendMetrics:
    """
    y_true / y_pred: array-like of trend labels (e.g. "growing", "stable",
    "receding", or compass-bin strings for direction). Multi-class, not
    just binary.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert len(y_true) == len(y_pred), "y_true and y_pred must be the same length"
    assert len(y_true) > 0, "cannot compute metrics on an empty set"

    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return TrendMetrics(
        n=len(y_true),
        accuracy=round(float(accuracy), 4),
        cohen_kappa=round(float(kappa), 4),
    )


def format_metrics_table(rows: list[dict], name_key: str = "name") -> str:
    """
    rows: list of dicts, each with `name_key` plus metric fields (from
    DetectionMetrics.as_dict() or TrendMetrics.as_dict(), merged with a
    name). Renders a simple fixed-width text table for README/terminal use.
    """
    if not rows:
        return "(no rows)"

    cols = [name_key] + [k for k in rows[0].keys() if k != name_key]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}

    def fmt_row(vals):
        return " | ".join(str(vals.get(c, "")).ljust(widths[c]) for c in cols)

    header = fmt_row({c: c for c in cols})
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(fmt_row(r) for r in rows)
    return f"{header}\n{sep}\n{body}"