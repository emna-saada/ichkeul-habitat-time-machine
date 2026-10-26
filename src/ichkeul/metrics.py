"""Segmentation metrics from a confusion matrix (rows = reference, cols = prediction)."""
from __future__ import annotations

import numpy as np

N_CLASSES = 8


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    m = (y_true >= 0) & (y_true < n)
    return np.bincount(n * y_true[m].astype(np.int64) + y_pred[m].astype(np.int64), minlength=n * n).reshape(n, n)


def scores(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    ref, pred = cm.sum(1), cm.sum(0)
    present = ref > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred > 0, tp / pred, 0.0)
        rec = np.where(ref > 0, tp / ref, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        iou = np.where(ref + pred - tp > 0, tp / (ref + pred - tp), 0.0)
    total = cm.sum()
    oa = tp.sum() / total
    pe = (ref * pred).sum() / total ** 2
    return {
        "overall_accuracy": float(oa),
        "kappa": float((oa - pe) / (1 - pe)) if pe < 1 else 0.0,
        "miou": float(iou[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class_f1": f1.round(4).tolist(),
        "per_class_iou": iou.round(4).tolist(),
        "per_class_precision": prec.round(4).tolist(),
        "per_class_recall": rec.round(4).tolist(),
        "support": ref.astype(int).tolist(),
        "confusion": cm.astype(int).tolist(),
    }
