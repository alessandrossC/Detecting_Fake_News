# 📰 Fake News Detection with BERT & Classical ML

This project solves the problem of **classifying news articles as fake (`is_fake=1`) or real (`is_fake=0`)** using both classical machine learning and state-of-the-art deep learning (BERT fine-tuning).

---

## 🚀 Project Overview

- **Goal:** Automatically identify fake news based on the title and text of news articles.
- **Dataset:** The dataset includes columns: `title`, `text`, `date`, and `is_fake`.
- **Approach:** The workflow combines data analysis, text preprocessing, feature engineering, classical ML (Logistic Regression, Random Forest, XGBoost), and fine-tuned BERT models.

---

## 📋 Workflow

1. **Exploratory Data Analysis (EDA)**
    - Analyze class distribution, text lengths, unique word counts, and common word frequencies.
2. **Text Preprocessing**
    - Tokenization, lowercasing, stopword removal, lemmatization/stemming.
3. **Feature Engineering**
    - Generate text-based features: length, punctuation counts, word overlaps, etc.
4. **Modeling**
    - Baseline models: `DummyClassifier`, Logistic Regression with TF-IDF/CountVectorizer.
    - Advanced models: `RandomForest`, `XGBoost`, BERT fine-tuning (Hugging Face Transformers).
5. **Evaluation**
    - Metrics: log loss, F1-score, accuracy on the validation set.
    - Visualization: calibration curves and feature importances.
6. **Model Interpretation**
    - Interpret BERT decisions using Integrated Gradients (Captum).
7. **Prediction (Inference)**
    - Use a helper function to predict fake/real news for any title and text.

---



