# LSTM & RNN-Based UPI Transaction Fraud Detection

> Research repository accompanying the paper submitted to the **2nd International Conference on Computational Intelligence and Information Retrieval (CIIR 2025)** — Springer Lecture Notes in Networks and Systems (LNNS), Scopus-indexed.

---

## 📌 Overview

This repository contains the complete research pipeline for detecting fraudulent UPI (Unified Payments Interface) transactions using recurrent neural network architectures. The work spans synthetic dataset generation, feature engineering, sequential deep learning model development, and comparison against gradient-boosted baselines — all under severe class imbalance conditions (~47:1 legitimate-to-fraud ratio).

---

## 🧠 Research Summary

**Problem:** UPI fraud detection is challenging due to extreme class imbalance, the sequential nature of user transaction behaviour, and the lack of large-scale real-world labelled datasets.

**Approach:** We model each user's transaction history as a temporal sequence and evaluate whether recurrent architectures (SimpleRNN, LSTM) can outperform standard gradient-boosted classifiers (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression) on imbalanced tabular-sequential data.

**Key Finding:** LSTM(64) achieves the best overall performance (ROC-AUC: 0.8368, PR-AUC: 0.4514, F1: 0.4684), outperforming the best SimpleRNN variant (ROC-AUC: 0.7828) and all gradient-boosted baselines on recall-sensitive metrics — demonstrating that gating mechanisms are necessary for effective sequential fraud modelling.

---

## 📂 Repository Structure
```
├── data_generation/          # Synthetic UPI transaction dataset generation scripts
├── feature_engineering/      # Feature construction (9 raw + 20 engineered features)
├── baselines/                # XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression
├── models/
│   ├── simplernn_64/         # SimpleRNN(64) + Dense(32) with Focal Loss
│   ├── simplernn_128/        # SimpleRNN(128) + Dense(32) with Focal Loss
│   └── lstm_64/              # LSTM(64) + Dense(32) with Focal Loss [best model]
├── discarded_experiments/    # Residual RNN and other failed architectures
├── results/                  # Evaluation metrics, ROC/PR curves, confusion matrices
├── paper/                    # Final submitted paper (PDF)
└── README.md
```

---

## 📊 Dataset

A synthetic dataset of **151,668 UPI transactions** was generated to simulate realistic payment behaviour across 1,000 users over a 3-month period (Jan–Mar 2025).

| Property | Value |
|---|---|
| Total Transactions | 151,668 |
| Unique Users | 1,000 |
| Fraud Cases | 3,131 (2.06%) |
| Legitimate Cases | 148,537 |
| Class Imbalance Ratio | ~47:1 |
| Date Range | Jan 1 – Mar 29, 2025 |
| Total Features Used | 29 (9 raw + 20 engineered) |

Key engineered features include temporal aggregates (`tx_count_last_5min`, `tx_count_last_1hr`), behavioural baselines (`amount_zscore`, `amount_vs_personal_mean`), geospatial signals (`is_location_jump`, `distance_from_prev_tx`), and device/network indicators (`isp_changed`).

---

## 🤖 Models & Results

### Sequential Deep Learning Models

All RNN/LSTM models use:
- Sequence length: 20 transactions per user
- Loss: Binary Focal Loss (α=0.75, γ=2.0)
- Optimiser: Adam (Kingma & Ba, 2015)
- Threshold: Tuned to maximise F1 on validation set

| Model | ROC-AUC | PR-AUC | F1 Score | Recall |
|---|---|---|---|---|
| SimpleRNN(64) | 0.7574 | 0.3492 | 0.4500 | 34.5% |
| SimpleRNN(128) | 0.7828 | 0.3556 | 0.4236 | — |
| **LSTM(64)** | **0.8368** | **0.4514** | **0.4684** | **38%** |

### Gradient-Boosted Baselines

| Model | ROC-AUC | F1 Score | Recall |
|---|---|---|---|
| Logistic Regression | — | 0.40 | 0.31 |
| Random Forest | — | 0.28 | 0.22 |
| LightGBM | — | 0.49 | 0.38 |
| XGBoost | — | 0.50 | 0.37 |
| CatBoost | — | 0.47 | 0.36 |

LSTM(64) achieves +7.0 percentage points improvement in ROC-AUC and +20.1 percentage points in PR-AUC over SimpleRNN(128), confirming that gating mechanisms are necessary for temporal fraud modelling under vanishing gradient conditions.

---

## 🔬 Discarded Experiments

A Residual RNN variant was explored but discarded due to training instability (F1 ≈ 0.05). This failure reinforces the paper's argument that skip connections alone are insufficient — proper gating (as in LSTM) is required for long-sequence fraud detection.

---

## ⚙️ Technical Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow / Keras
- **ML Baselines:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Environment:** Google Colab (GPU)
- **Data Handling:** pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn

---

## 📄 Paper

**Title:** *Sequential Deep Learning for UPI Transaction Fraud Detection under Class Imbalance*

**Authors:** Santwan *(first author)*, Avhik Laha, Debasri Pal, Debadrita Basu, Rima Pahari

**Supervisor:** Prof. Anupam Ghosh

**Venue:** 2nd International Conference on Computational Intelligence and Information Retrieval (CIIR 2025), Springer LNNS (Scopus-indexed)

**Submission ID:** 708

The paper PDF is available in the `/paper` directory of this repository.

---

## 📋 Citation

If you use this work, please cite:
```bibtex
@inproceedings{santwan2025upi,
  title     = {Sequential Deep Learning for UPI Transaction Fraud Detection under Class Imbalance},
  author    = {Santwan and Laha, Avhik and Pal, Debasri and Basu, Debadrita and Pahari, Rima},
  booktitle = {Proceedings of the 2nd International Conference on Computational Intelligence and Information Retrieval (CIIR)},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  year      = {2025}
}
```

---

## 📬 Contact

For questions related to the dataset, models, or paper, please open an issue in this repository.


## License

**Code** (data generation, feature engineering, model notebooks): [MIT License](./LICENSE)

**Paper content** (PDF, figures, written text in `/paper`): 
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

© 2025 Santwan et al. The paper is submitted to Springer LNNS (CIIR 2025). 
Reproduction or adaptation of the paper content without permission is prohibited.
