# Fraud Detection Notebook – Tabular + Graph Approaches

Comparison of classical ML and graph-based methods for detecting fraudulent / illicit transactions.

## Overview

This notebook explores two very different fraud detection settings:

1. **Credit Card Transactions** (tabular, highly imbalanced)
   - Kaggle Credit Card Fraud dataset (~284k rows, 0.17% fraud)
   - Focus: handling extreme class imbalance, PR-AUC evaluation
   - Models: Random Forest + SMOTE vs XGBoost with scale_pos_weight

2. **Bitcoin Transaction Graph** (Elliptic dataset)
   - ~204k transactions, ~2% illicit labels, large portion unlabeled
   - Graph structure: payment flows between addresses/transactions
   - Evolution: GCN → GraphSAGE → Hybrid GraphSAGE embeddings + XGBoost
   - Best result: AUPRC ≈ 0.75 (competitive with published SOTA on original Elliptic)

## Key Results

| Dataset              | Model                          | Main Metric | Value   |
|----------------------|--------------------------------|-------------|---------|
| Credit Card Fraud    | XGBoost + class weighting      | AUPRC       | ~0.878  |
| Credit Card Fraud    | Random Forest + SMOTE          | AUPRC       | ~0.829  |
| Elliptic (Bitcoin)   | Vanilla GCN                    | AUPRC       | ~0.545  |
| Elliptic (Bitcoin)   | GraphSAGE                      | AUPRC       | ~0.665  |
| Elliptic (Bitcoin)   | GraphSAGE embeddings + XGBoost | AUPRC       | **~0.754** |

## What You Will Find in the Notebook

- Standard tabular fraud pipeline (SMOTE, class weighting, PR curves)
- Full PyG implementation of GCN and GraphSAGE on Elliptic
- Hybrid approach: using GraphSAGE node embeddings + original features → XGBoost
- t-SNE visualization of learned GraphSAGE latent space (clear licit vs illicit separation)
- Emphasis on **AUPRC** as the primary metric due to severe imbalance
