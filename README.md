# Fake Review Detection using NLP & Machine Learning

This project focuses on detecting **fake vs genuine product reviews** using Natural Language Processing (NLP), traditional Machine Learning models, and Deep Learning approaches.

The goal is to compare multiple models and identify the best-performing approach for fake review classification.

---

##  Project Overview

Online platforms suffer from fake reviews that mislead customers.  
This project builds and evaluates multiple models to classify reviews as **Fake** or **Genuine**.

We implemented:

- Text preprocessing using NLTK
- TF-IDF feature extraction
- Traditional ML models
- Deep learning model (GRU)
- Transformer embeddings (DistilBERT)

---

##  Tech Stack

- Python
- Pandas & NumPy
- NLTK (Text preprocessing)
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- PyTorch
- HuggingFace Transformers (DistilBERT)
- Matplotlib & Seaborn

---

##  Models Implemented

###  Traditional ML Models
- Decision Tree (with GridSearchCV)
- Random Forest (with GridSearchCV)
- XGBoost (with GridSearchCV)

### Deep Learning
- GRU (Gated Recurrent Unit)

### Transformer-Based Approach
- DistilBERT embeddings
- Neural classifier built on embeddings

---

## Features

- Text cleaning (lowercasing, stopword removal, lemmatization)
- TF-IDF vectorization
- Hyperparameter tuning using GridSearchCV
- Model comparison
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - Classification Report

---

## 📂 Repository Structure
