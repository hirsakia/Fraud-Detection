# Multi-Domain Fraud Detection

Three fraud detection pipelines across different domains and scales — from tabular credit card transactions to graph-structured blockchain forensics. Each notebook is self-contained.

## Datasets & Results

### 1. Credit Card Fraud (`FraudDet_CCF.ipynb`)

**Data:** 284,807 transactions, 0.17% fraud rate (Kaggle).

| Model | Approach | AUPRC |
|:---|:---|:---|
| Random Forest + SMOTE | Oversample minority, then classify | 0.8293 |
| **XGBoost (cost-sensitive)** | `scale_pos_weight` on raw data | **0.8779** |
| Isolation Forest | Unsupervised anomaly detection | — |
| Autoencoder | Reconstruction error on normal-only training | — |

**Key finding:** Cost-sensitive weighting on the native imbalanced distribution outperforms synthetic oversampling — SMOTE introduces noise at a 0.17% fraud rate. Cost-sensitive threshold optimization (FN=100, FP=10) further improves operational performance.

### 2. Bank Account Fraud (`FraudDet_BAF.ipynb`)

**Data:** ~1M accounts from the NeurIPS 2022 BAF benchmark suite (`Base.csv`).

| Step | Detail |
|:---|:---|
| Split | Temporal train/test to prevent leakage |
| Encoding | Dummy variables for categorical features |
| Model | XGBoost with `scale_pos_weight` |
| Fairness | Age-group FPR audit (disparity range: 0.03–0.24) |

**Key finding:** Temporal splitting matters — random splits inflate metrics by leaking future patterns into training. The fairness audit reveals non-trivial FPR disparity across age bins, flagging potential regulatory concerns.

### 3. Elliptic Bitcoin (`FraudDet_Elliptic.ipynb`)

**Data:** 203,769 nodes, 234,355 edges, 165 features per transaction, 77% unlabeled (Elliptic dataset via PyG).

| Model | Methodology | AUPRC |
|:---|:---|:---|
| Vanilla GCN | Standard message passing | 0.5453 |
| GraphSAGE | Inductive neighborhood sampling | 0.6650 |
| **Hybrid (SAGE + XGB)** | **GNN embeddings + raw features → XGBoost** | **0.7539** |

**Architecture evolution:**

1. **GCN → GraphSAGE (+22%):** Inductive neighborhood sampling focuses on local structural signatures (peeling chains, rapid distribution patterns) instead of noisy global connectivity.
2. **GraphSAGE → Hybrid (+16%):** 64-dim SAGE embeddings are stacked with raw transaction features and fed to XGBoost. The gradient-boosted classifier resolves ambiguous nodes that neither relational nor tabular models handle alone.
3. **t-SNE visualization** confirms separable "Fraud Islands" on the periphery of the licit cluster, validating that the GNN latent space captures money-laundering topology.

## Stack

- **Tabular ML:** XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Deep Learning:** TensorFlow/Keras (autoencoder)
- **Graph Neural Networks:** PyTorch Geometric (GCN, GraphSAGE)
- **Visualization:** matplotlib, t-SNE (scikit-learn)
- **Data:** pandas, NumPy

## Takeaways

- **Class imbalance:** Cost-sensitive learning > synthetic oversampling at extreme imbalance ratios.
- **Graph structure matters:** When transaction topology is available, GNN embeddings add signal that flat features miss.
- **Hybrid stacking:** The best results come from combining relational context (GNN embeddings) with discriminative raw features in a gradient-boosted classifier.
- **Fairness is not optional:** Even well-performing models can exhibit demographic FPR disparity — always audit.
