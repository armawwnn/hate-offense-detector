# Hate-Speech & Offensive-Language Tweet Classifier  
*A collection of classic discriminant models implemented from scratch*

---

## 📑 Project Overview
This repository contains a complete, end-to-end pipeline that classifies English tweets into  

| Label | Meaning                |
|:----:|-------------------------|
| **0** | Hate Speech            |
| **1** | Offensive Language     |
| **2** | Neither (Harmless)     |

Six traditional statistical classifiers are coded **manually**, without relying on high-level `sklearn` discriminant classes:

| Model | Key Assumption |
|-------|----------------|
| **Linear Discriminant Analysis (LDA)** | shared full covariance matrix |
| **Quadratic Discriminant Analysis (QDA)** | class-specific full covariance |
| **Diagonal Gaussian (≈ Gaussian Naïve Bayes)** | diagonal covariance |
| **Spherical Gaussian** | single variance \(s^2\) for all features |
| **Nearest Mean Classifier (NMC)** | Euclidean distance to class means |
| **Template Matching** | dot product \( g_i(x)=\mu_i^\top x \) |

Every model is evaluated with **10-fold stratified cross-validation** and detailed metrics.

---

## 🔢 Dataset
We use the public **“Hate Speech and Offensive Language”** corpus collected by  
*Thomas Davidson et al., 2017*  
<https://github.com/t-davidson/hate-speech-and-offensive-language>

Each record provides:
- the raw tweet text  
- three vote counts (`hate_speech`, `offensive_language`, `neither`)  
- a majority-label column `class ∈ {0, 1, 2}`  

---

## 🏗️ Pipeline
```text
Tweet → Clean text → TF-IDF (5 000) → Truncated SVD (100)  \
                              ↘ optional scaled vote counts ↗
                         final feature vector (100 – 103 dims)
