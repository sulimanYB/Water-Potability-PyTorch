import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, data_loader, threshold=0.5):
    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for features, labels in data_loader:
            logits = model(features)
            probabilities = torch.sigmoid(logits).view(-1)

            all_labels.extend(labels.numpy().astype(int))
            all_probabilities.extend(probabilities.numpy())

    # Convert probabilities to predicted classes
    predictions = [
        int(p >= threshold) for p in all_probabilities
    ]

    # Calculate metrics
    results = {
        "accuracy": accuracy_score(all_labels, predictions),
        "precision": precision_score(
            all_labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            all_labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            all_labels, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            all_labels, predictions,
            average="macro", zero_division=0
        ),
        "roc_auc": roc_auc_score(
            all_labels, all_probabilities
        ),
        "confusion_matrix": confusion_matrix(
            all_labels, predictions
        )
    }

    print(classification_report(
        all_labels,
        predictions,
        target_names=["Not Potable", "Potable"],
        zero_division=0
    ))

    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    return results