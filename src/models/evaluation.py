from typing import Dict, Sequence, Union

from sklearn.metrics import confusion_matrix, precision_score, recall_score


def calculate_threshold_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> Dict[str, Union[float, int]]:
    """Calculate classification tradeoffs for a probability threshold."""
    predictions = [int(probability > threshold) for probability in probabilities]
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    )

    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives),
    }