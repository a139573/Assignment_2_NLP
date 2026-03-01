# Neural Models for Human Values Detection
**Natural Language Processing (2025/2026) — Assignment 2** **Authors:** Roberto Aldanondo & Andrés Malón  

## Overview & Task
This repository contains the neural sequence modeling track for the multi-label classification of 20 human values (derived from the Schwartz Basic Human Values model). The task requires mapping argumentative texts—composed of a **Conclusion + Stance + Premise**—to one or more binary value labels.

[Image of multi-label text classification machine learning pipeline]

## Dataset & The Imbalance Challenge
* **Source**: Touché Task-4 Human Values Detection.
* **Size**: 8,865 examples across 20 labels.
* **Split**: 80% train / 10% val / 10% test (using iterative multilabel stratification).

A core challenge of this dataset is extreme class imbalance, where minority classes comprise <5% of the data. Initial neural models suffered a majority-class collapse (F1-Macro of 0.00). This was mitigated using a mathematically dampened Positive Weight ($w_{pos}$) array in the Binary Cross-Entropy loss function, forcing the network to prioritize rare labels.

## Models Evaluated
1. **Baseline (Assignment 1)**: `CountVectorizer(1,3)` + OneVsRest Logistic Regression.
2. **BiLSTM**: 2-layer bidirectional LSTM trained from scratch with Global Max Pooling.
3. **Transformers**: Fine-tuned variants including DistilBERT (`distilbert-base-uncased`), BERT, RoBERTa, and FinBERT (for domain-mismatch experimentation).

## Repository Structure
```text
assignment2/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Original TSV files (git-ignored)
│   └── processed/            # Stratified 80/10/10 splits (git-ignored)
├── models/
│   ├── lstm.py               # Custom BiLSTM architecture
│   ├── transformer.py        # Universal Hugging Face AutoModel wrapper
│   └── saved_weights/        # Trained .pt files (git-ignored)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_error_analysis.ipynb
└── results/                  # All plots, CSVs, and qualitative analysis
    ├── architecture_comparison.csv
    ├── architecture_comparison_plots.png
    ├── per_label_f1_heatmap.png
    └── attention_visualizations/
```

## Setup & Execution
1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Place the original dataset .tsv files into data/raw/.

3. Execute the notebooks sequentially in notebooks/ to reproduce the entire pipeline from text cleaning to attention visualization.

## Systematic Experiments
The notebooks collectively execute the five required systematic experiments:

1. Architecture Comparison: Evaluation of F1-Macro, parameter count, training time, and GPU memory across all models.

2. Learning Curves: Performance benchmarking at 25%, 50%, 75%, and 100% training data subsets.

3. Ablation Studies: Evaluation of BiLSTM dimensions (directionality, layers, hidden dim) and Transformer configurations (frozen vs. unfrozen encoders, learning rates).

4. Error Analysis: Isolation of errors fixed/introduced by neural models compared to the baseline, including per-label heatmaps and layer-attention visualization.

5. Computational Cost: Milliseconds-per-sample inference tracking and deployment trade-off discussions.

Reproducibility
- Random Seed: 42 enforced across PyTorch, NumPy, and Scikit-Learn.

- Early Stopping: patience=5 for LSTM, patience=3 for Transformers.

- Gradient Clipping: max_norm=1.0.

- Threshold Optimization: Decision thresholds are dynamically optimized (between 0.1 and 0.9) per label on the validation set to maximize final Test F1-Macro.