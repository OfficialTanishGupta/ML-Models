# 📧 Logistic Regression — Spam Detection

A machine learning project that classifies SMS messages as **Spam or Not Spam** using Logistic Regression and TF-IDF.

## 🚀 Features

- Text preprocessing
- TF-IDF vectorization
- Logistic Regression classifier
- Model saving/loading
- Command-line prediction script

## 📂 Project Structure

```text
Logistic-Regression-Spam-Detection/
├── data/               # Raw and processed datasets (e.g., SMSSpamCollection)
├── notebook/           # Jupyter notebooks for EDA and prototyping
├── src/                # Source code for data cleaning, features, and training
├── model/              # Saved model binaries (e.g., .pkl or .joblib files)
├── requirements.txt    # List of Python dependencies
└── README.md           # Project documentation and setup instructions
```

## ⚙️ How to Run

### 1️⃣ Install dependencies

pip install -r requirements.txt

### 2️⃣ Train model

python src/train.py

### 3️⃣ Predict new message

python src/predict.py

## 🎯 Output

Classifies messages as:

- 🚨 Spam
- ✅ Not Spam

## 🧠 Model Used

- Logistic Regression
- TF-IDF Text Vectorization

## 📊 Dataset

SMS Spam Collection Dataset (UCI/Kaggle)

---

⭐ Ideal beginner NLP project for portfolios and interviews.
