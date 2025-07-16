# 🩺 Breast Cancer Classification using Machine Learning

## 🎯 Project Overview
Developed a robust **machine learning classification system** to predict whether breast cancer tumors are **benign** or **malignant** using **clinical biopsy features**.

### 📌 Target Variable:
- **Diagnosis (Target):** Binary classification:
    - `0` → **Benign (Non-cancerous)**
    - `1` → **Malignant (Cancerous)**

### 🎁 Input Features:
- **30 Numerical Features** extracted from **biopsy test data**, including:
    - Radius, Texture, Perimeter, Area, Smoothness
    - Compactness, Concavity, Concave points, Symmetry
    - Fractal Dimension (mean, standard error, worst)

### 🏷️ Goal:
- To predict the **diagnosis label (Malignant or Benign)** using machine learning techniques for **early detection of breast cancer**.

---

## ✅ Objective
- Built a complete **end-to-end ML pipeline** for **early-stage breast cancer detection** using structured clinical data.
- Focused on **optimizing accuracy**, **minimizing false positives**, and improving **model explainability**.


## 📊 Key Highlights

- **Dataset:** [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Techniques Applied:**
    - Data Cleaning & Preprocessing (handling missing values, feature scaling)
    - Exploratory Data Analysis (EDA)
    - Supervised Learning Classification Models
- **Algorithms Used:**
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Naive Bayes
    - Support Vector Machine (SVM)
    - Decision Tree Classifier
    - Random Forest Classifier
    - Gradient Boosting Classifier
- **Tools & Libraries:** Python, pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 🏆 Performance Summary

| Model                        | Accuracy |
|------------------------------|----------|
| Support Vector Machine (SVM) | 98% ✅    |
| Logistic Regression          | 97%      |
| Naive Bayes                  | 97%      |
| Random Forest Classifier     | 96%      |
| Gradient Boosting Classifier | 96%      |
| K-Nearest Neighbors (KNN)    | 95%      |
| Decision Tree Classifier     | 94%      |

- 🚀 **SVM achieved the highest accuracy of 98%.**
- Evaluated using **Confusion Matrix**, **Classification Report**, and **ROC-AUC curves**.

---

## 💡 Outcome & Skills Gained
- Hands-on experience in **data preprocessing**, **EDA**, and **classification modeling**.
- Advanced **Python programming skills** with pandas, seaborn, and scikit-learn.
- Practical understanding of **healthcare data analytics**.
- Learned to **compare multiple models**, **evaluate performance**, and **visualize results**.



## 🚀 Future Scope
- ✅ Hyperparameter Tuning (GridSearchCV)
- ✅ Deploy model as a **Flask/Streamlit web app**
- ✅ Dockerize project for easy deployment

---

## ✨ How to Run

```bash
pip install -r requirements.txt
python breast_cancer_classification.py

---

## 📫 Project Credits
👤 **Project done by G Pragnya Reddy of 2nd year CS-DS on 4th April, 2024**

---
