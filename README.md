# Credit Card Fraud Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arda-duman/credit-card-fraud-detection/blob/main/fraud_detection.ipynb)

A machine learning model that flags fraudulent credit card transactions. The project is a case study in handling extreme class imbalance and choosing evaluation metrics that actually matter when only 0.17% of the data is the class you care about.

## Results

Final model: **XGBoost at a 0.5 decision threshold**, evaluated once on a held-out test set of 56,962 transactions (98 frauds) that was never used for training, model selection or threshold tuning.

| Metric | Fraud class |
|---|---|
| Precision | 89% |
| Recall | 83% (81 of 98 frauds caught) |
| F1-score | 0.86 |
| False alarms | 10 out of 56,864 legitimate transactions |

Confusion matrix:

```
                 Predicted legit   Predicted fraud
Actual legit          56,854             10
Actual fraud              17             81
```

Accuracy is not reported as a headline metric: a model that predicts "legit" for every transaction would score 99.83% accuracy while catching zero fraud.

## Model comparison

All three models were trained on the same data with class weighting and compared on the validation set at the default 0.5 threshold:

| Model | Precision | Recall | False alarms |
|---|---|---|---|
| Logistic Regression | 6% | 90% | 1,403 |
| Random Forest | 96% | 71% | 3 |
| **XGBoost** | **92%** | **78%** | **7** |

Logistic Regression catches the most fraud but buries it in false alarms. Random Forest is the most precise but misses too much fraud. XGBoost gives the best balance.

## Threshold tuning

Lowering the decision threshold trades false alarms for recall. On the validation set:

| Threshold | Recall | Precision | False alarms |
|---|---|---|---|
| 0.5 | 78% | 92% | 7 |
| 0.1 | 78% | 85% | 14 |
| 0.01 | 81% | 68% | 38 |
| 0.005 | 83% | 59% | 56 |

Going from 0.5 to 0.005 catches only about 5 more frauds while adding about 50 false alarms. XGBoost's scores are already well separated, so the default threshold was kept. A bank that weighs missed fraud far more heavily than review costs could reasonably choose a lower threshold.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on Kaggle (ULB Machine Learning Group).

- 284,807 transactions, 492 of them fraudulent (0.17%)
- Features `V1` to `V28` are PCA-transformed for confidentiality, plus `Time` and `Amount`
- Target column: `Class` (1 = fraud, 0 = legitimate)

The notebook downloads the dataset automatically with `kagglehub`, so it is not stored in this repository.

## Approach

1. **Explore** the data and quantify the class imbalance
2. **Split** the data 60/20/20 into train, validation and test sets, stratified to preserve the fraud ratio
3. **Scale** features with a scaler fit on the training set only
4. **Handle imbalance** with class weighting (`class_weight='balanced'`, and `scale_pos_weight ≈ 577` for XGBoost) rather than resampling
5. **Compare models** (Logistic Regression, Random Forest, XGBoost) on the validation set
6. **Tune the decision threshold** on the validation set using a precision/recall curve
7. **Evaluate once** on the untouched test set

## How to run

### In Google Colab (easiest)
Click the **Open in Colab** badge above, then **Runtime > Run all**. The dataset downloads automatically on the first run.

### Locally
```bash
git clone https://github.com/YOUR-USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, matplotlib, kagglehub

## Author

Arda Duman
