from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


def save_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    figure, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred))
    disp.plot(ax=ax)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def save_roc_curve(y_true: Sequence[int], y_prob: Sequence[float], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    figure, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
