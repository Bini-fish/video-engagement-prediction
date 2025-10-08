# Predicting and Understanding Viewer Engagement with Educational Videos

## 1. Introduction

Online education has revolutionized how people learn. With open platforms like [videolectures.net](http://videolectures.net/) and MOOCs such as Coursera, millions of learners now have access to an enormous library of lectures and tutorials.  
But this abundance of content brings a new challenge: **how do we find and recommend videos that keep learners engaged?**

In this project, I explore how **machine learning** can be used to predict how engaging an educational video will be — before anyone even watches it.

---

## 2. The Challenge

Engagement is one of the most important qualities of a good learning video.  
If a video fails to capture attention, viewers typically stop watching within the first few minutes.

To address this, I framed the problem as a **binary classification task**:  
> “Will a given educational video be engaging — meaning at least 30% of it is watched by most viewers — or not?”

By analyzing text, audio, and structural features of videos, we aim to predict engagement and uncover what makes learning content successful.

---

## 3. The Dataset

The dataset comes from the **VLE Dataset** created by [Sahan Bulathwela (University College London)](https://github.com/sahanbull/VLE-Dataset), containing metadata and derived features from educational videos.

Each record represents a single video and includes numerical features capturing its content and delivery style.

### Key Features

| Feature | Description |
|----------|--------------|
| `title_word_count` | Number of words in the video’s title |
| `document_entropy` | How varied the transcript topics are (lower = more focused) |
| `freshness` | Days since 01/01/1970 (newer videos have higher values) |
| `easiness` | Readability score of the transcript (lower = more complex) |
| `fraction_stopword_presence` | Fraction of stopwords like “the” and “and” in the transcript |
| `speaker_speed` | Words per minute spoken by the presenter |
| `silent_period_rate` | Fraction of time with silence during the video |

The **target variable** is `engagement`, which is **True** if the median viewer watched ≥30% of the video, and **False** otherwise.

## Data files

>assets/train.csv # Training set with engagement labels 

>assets/test.csv # Test set (no labels)


---

## 4. Modeling Approach

The project follows a structured ML pipeline:

1. **Data Preparation**
   - Load and inspect CSVs
   - Handle missing values
   - Scale features using `StandardScaler` to normalize feature ranges

2. **Model Selection**
   - Tested multiple models:
     - Logistic Regression  
     - Random Forest  
     - Gradient Boosting (GBDT)

3. **Hyperparameter Tuning**
   - Used **GridSearchCV** with **Stratified K-Fold** cross-validation  
   - Optimized models for the **ROC-AUC** metric

4. **Model Evaluation**
   - Measured AUC for each model  
   - Selected the best performer  
   - Visualized feature impact and ROC curves

---

## 5. Evaluation Metric

Performance was assessed using **Area Under the ROC Curve (AUC)**:  
- AUC ≥ 0.80 → good predictive ability  
- AUC ≥ 0.85 → excellent performance

Each model outputs a probability that a video will be engaging.  
Predictions are stored as:

```python
id
9240    0.401958
9241    0.105928
9242    0.018572
...
```
## 6. Visualizations

Two main plots help interpret results:

* **AUC Bar Chart** — compares model performance
* **ROC Curve** — visualizes trade-offs between sensitivity and specificity

All generated under `results/`:

results/
 submission.csv, auc_bar_chart.png, roc_comparison.png

---
## 7. Project Structure

video-engagement-prediction/
  - assets/
    - train.csv
    - test.csv
  - src/
    - engagement_model.py
    - visualization.py
  - results/                # Auto-generated output folder
  - main.py                 # Project runner
  - requirements.txt        # Dependencies
  - README.md               # Project overview

## 8. Code Snippets

### Model Training and Evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('assets/train.csv')
X = df.drop(columns=['engagement'])
y = df['engagement']

# Split and scale
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, preds)
    results[name] = auc

# Plot AUC comparison
plt.bar(results.keys(), results.values())
plt.title('Model AUC Comparison')
plt.ylabel('AUC')
plt.savefig('results/auc_bar_chart.png')
plt.show()
```

## ROC Curve Visualization

```python
plt.figure()
for name, model in models.items():
    preds = model.predict_proba(X_val_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, preds)
    plt.plot(fpr, tpr, label=f'{name} (AUC={results[name]:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('results/roc_comparison.png')
plt.show()
```
## 9. How to Run the Project

### Clone the repository

```python
git clone https://github.com/yourusername/video-engagement-prediction.git
cd video-engagement-prediction
```
## Install dependencies

```python
pip install -r requirements.txt
```

### Run the main script
```python
python main.py
```
## View results

- Predictions: results/submission.csv
- Charts: results/auc_bar_chart.png, results/roc_comparison.png

## 10. Insights and Extensions

This project demonstrates how machine learning can guide the creation and recommendation of more engaging educational videos.
Future extensions could include:

- Natural Language Processing (NLP) to analyze full transcripts

- Deep learning models (e.g., XGBoost, LightGBM, or Transformers)

- Feature importance explainability with SHAP or LIME

## 11. Credits

Dataset originally developed by Sahan Bulathwela at University College London
Project concept inspired by Coursera’s Machine Learning with Python course.