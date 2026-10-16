from src.models.evaluation import calculate_threshold_metrics


def test_calculate_threshold_metrics_uses_strict_threshold() -> None:
    metrics = calculate_threshold_metrics(
        y_true=[1, 1, 0, 0],
        probabilities=[0.9, 0.5, 0.6, 0.2],
        threshold=0.5,
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["true_negatives"] == 1


def test_calculate_threshold_metrics_handles_no_positive_predictions() -> None:
    metrics = calculate_threshold_metrics(
        y_true=[1, 0], probabilities=[0.4, 0.3], threshold=0.5
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["false_negatives"] == 1