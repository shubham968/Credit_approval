# Credit Approval Machine Learning Project

This project builds a machine learning model to predict whether a credit application should be approved or not. The model is trained on the UCI Credit Approval dataset, which includes both categorical and numerical features, along with missing values, outliers, and skewed distributions.

---

## 📂 Dataset

- Source: [UCI ML Repository – Credit Approval Dataset](https://archive.ics.uci.edu/dataset/27/credit+approval)
- Contains mixed feature types (categorical + numerical)
- Real-world issues like missing values, imbalanced categories, and outliers

---

## ✅ Key Highlights

- 🔄 Fetched live data using `ucimlrepo`
- 🧼 **Data Preprocessing**:
  - Handled missing values using mode and median
  - Replaced rare categorical values with `'Other'`
  - Detected and clipped outliers using IQR
  - Log-transformed skewed numeric columns
  - Dropped noisy feature `A15` after boxplot inspection
- 🧠 **Feature Encoding**:
  - Used `TargetEncoder` for categorical features
  - Prevented data leakage by fitting encoders only on training data
- 🧪 **Modeling**:
  - Tried multiple classifiers: Logistic Regression, Random Forest, XGBoost, etc.
  - Evaluated using Accuracy, F1 Score, and ROC-AUC
- 📊 **Best Model**: Logistic Regression with
  - F1 Score ≈ **0.85**
  - ROC-AUC ≈ **0.93**
  - Accuracy ≈ **88%**

---

## 📈 Evaluation Metrics (5-Fold Cross-Validation)

| Model               | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | ~88%     | ~0.85    | ~0.93   |
| Random Forest       | ~87%     | ~0.84    | ~0.92   |
| XGBoost             | ~87%     | ~0.84    | ~0.92   |

---

