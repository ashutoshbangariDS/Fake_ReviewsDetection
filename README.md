# Fake Review Detection using NLP, GRU & Transformer Models

This project focuses on detecting **Fake vs Genuine product reviews** using Natural Language Processing (NLP), Deep Learning, and Transformer-based embeddings.

The objective is to compare a baseline GRU model with a Transformer-enhanced GRU model and evaluate performance improvements.

---

##  Project Overview

Online platforms often suffer from fake reviews that mislead customers.  
This project builds an end-to-end text classification pipeline to detect fraudulent reviews using:

- Text preprocessing (NLTK)
- Deep learning with GRU
- Transformer embeddings (DistilBERT)
- Model comparison using evaluation metrics

---

##  Dataset Analysis

The dataset contains approximately:

- Genuine Reviews (CG): ~20000,200
- Fake Reviews (OR): ~20000,300
- Total Samples: ~40000,500

The dataset is well-balanced (~50:50 distribution), ensuring unbiased model training and reliable evaluation.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- HuggingFace Transformers (DistilBERT)
- Matplotlib & Seaborn

---

# 📈 Model Performance

## 🔹 1. GRU Model (Baseline)

**Test Accuracy: 83.36%**

### Confusion Matrix

|               | Predicted CG | Predicted OR |
|---------------|-------------|-------------|
| Actual CG     | 2935        | 1101        |
| Actual OR     | 244         | 3803        |

### Observations
- Strong detection of Fake reviews
- Higher misclassification of Genuine reviews
- Moderate class imbalance in predictions

---

## 🔹 2. DistilBERT + GRU Model (Best Model)

**Test Accuracy: 86%**  
Macro F1 Score: 0.86  
Weighted F1 Score: 0.86  
Test Samples: 8,083  

### Confusion Matrix

|               | Predicted CG | Predicted OR |
|---------------|-------------|-------------|
| Actual CG     | 3507        | 529         |
| Actual OR     | 593         | 3454        |

### Observations
- Balanced classification performance
- Reduced false positives and false negatives
- ~3% improvement over baseline GRU
- Better contextual understanding using Transformer embeddings

🏆 **Best Performing Model: DistilBERT + GRU**

---

## 📉 Training Behavior

- Final Training Accuracy: ~86.7%
- Final Validation Accuracy: ~86.1%
- Minimal overfitting observed
- Stable convergence after ~10 epochs

The close alignment between training and validation curves indicates strong generalization.

---



#  How to Run

## Option 1: Run Locally

## Clone the Repository

```bash
git clone https://github.com/ashutoshbangariDS/Fake_ReviewsDetection.git
cd Fake_ReviewsDetection
```

###  Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Run Notebook

```bash
jupyter notebook
```

Open:
```
Ashuthosh_Fake_Review_Code.ipynb
```

Run all cells sequentially.

---

## 🔹 Option 2: Run in Google Colab

1. Upload the notebook
2. Upload the dataset
3. Update dataset path if required
4. Run all cells

---

# 🔍 Key Features

- Text preprocessing (stopword removal, lemmatization)
- Deep learning with GRU
- Transformer embeddings using DistilBERT
- Model comparison and evaluation
- Confusion matrix visualization
- Training vs Validation performance tracking

---

# 🎯 Key Learnings

- End-to-end NLP pipeline development
- Sequence modeling using GRU
- Contextual embeddings with Transformers
- Model evaluation and error analysis
- Performance comparison between baseline and enhanced architectures

---


⭐ If you found this project interesting, feel free to star the repository!
