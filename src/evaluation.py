import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

def get_predictions(model, loader, device):
    """Generates predictions and probabilities for a given dataset."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()

def generate_classification_report(labels, preds, class_names):
    """Prints overall accuracy, class-wise accuracy, and the sklearn classification report."""
    print("--- Classification Report ---")
    print(classification_report(labels, preds, target_names=class_names))
    
    cm = confusion_matrix(labels, preds)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("--- Per-Class Accuracy ---")
    for i, acc in enumerate(class_accuracy):
        print(f"{class_names[i]} Accuracy: {acc*100:.2f}%")

def plot_confusion_matrix(labels, preds, class_names, save_path=None):
    """Plots and optionally saves the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_roc_curve(labels, probs, class_names, save_path=None):
    """Plots multi-class ROC curves with AUC."""
    labels_bin = label_binarize(labels, classes=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multi-Class)")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_pr_curve(labels, probs, class_names, save_path=None):
    """Plots multi-class Precision-Recall curves."""
    labels_bin = label_binarize(labels, classes=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))

    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        ap = average_precision_score(labels_bin[:, i], probs[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP = {ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Multi-Class)")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()