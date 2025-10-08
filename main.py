"""
main.py — One-click runner for the Educational Video Engagement Prediction project.
"""

import os
from src.engagement_model import engagement_model
from src.visualization import plot_auc_bar_chart, visualize_model_performance

def main():
    # Create results folder if not exists
    os.makedirs("results", exist_ok=True)

    print("Training models and evaluating performance...\n")
    results, model_auc_scores, best_model, X_train_scaled, y_train = engagement_model()

    print("\n Model training complete. Generating visualizations...\n")

    # Visualizations
    plot_auc_bar_chart(model_auc_scores)
    visualize_model_performance({f"BestModel": best_model}, X_train_scaled, y_train)

    print("\n All done! Check the 'results/' folder for outputs:")
    print(" - submission.csv")
    print(" - auc_bar_chart.png")
    print(" - roc_comparison.png")

if __name__ == "__main__":
    main()
