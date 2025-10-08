import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

def visualize_model_performance(models, X_train_scaled, y_train):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_proba = model.predict_proba(X_train_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_proba)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {model_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/roc_comparison.png", dpi=300)
    plt.show()

def plot_auc_bar_chart(model_auc_scores):
    """Bar chart comparing mean AUC scores of models."""
    plt.figure(figsize=(6, 4))
    names = list(model_auc_scores.keys())
    scores = list(model_auc_scores.values())
    bars = plt.bar(names, scores, color="#4C72B0")

    plt.title("Model AUC Comparison")
    plt.ylabel("Mean AUC (Cross-Validation)")
    plt.ylim(0.7, 1.0)
    plt.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, score + 0.01,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/auc_bar_chart.png", dpi=300)
    plt.show()
