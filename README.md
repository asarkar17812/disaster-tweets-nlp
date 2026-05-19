# Kaggle Disaster Tweets Competition

This repo contains a project I started over a year ago and am now revisiting as part of an Intro to Machine Learning class. The original version was a soft-voting ensemble of an LSTM, BERT (cased), and RoBERTa. The current version keeps the soft-voting idea but replaces BERT with [BERTweet](https://huggingface.co/vinai/bertweet-base) (Twitter-pretrained), upgrades the LSTM with attention pooling and multi-sample dropout, integrates the Kaggle `keyword` column as a topic prefix, and adds a held-out test split + a grid search over per-model hyperparameters and ensemble weights.

## Learning outcomes

1. Understanding transformers, tokenization, and encoding
2. Understanding ensemble models and improving prediction accuracy by combining individual models
3. Understanding GPU acceleration and passing data as tensors for improved training speed
4. Classification tasks and NLP as a linguistic science
5. Optimization using AdamW (Adaptive Moment Estimation with Weight Decay)
6. Layer-wise learning-rate decay for transformer fine-tuning
7. Attention pooling and multi-sample dropout as regularizers for sequence classifiers
8. Honest evaluation — held-out test sets, val-only threshold/weight tuning, stratified splits
9. Hyperparameter search via grid over a validation set

## Model architecture

The ensemble has three soft-voting members. Each member produces a class-1 probability, the three probabilities are combined into a single probability via val-tuned weights, and a val-tuned threshold turns that into a 0/1 prediction.

| Model              | Checkpoint                                          | Why it's in the ensemble                                                                                         |
| ------------------ | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **BERTweet** | `vinai/bertweet-base`                             | RoBERTa pretrained on ~850M tweets — domain-matched to our data.                                                |
| **RoBERTa**  | `roberta-base`                                    | General-domain RoBERTa; errors only partially correlated with BERTweet's, which is what makes the ensemble work. |
| **BiLSTM**   | trained from scratch with BERT-cased embedding init | Adds further diversity; uses attention pooling over time and a multi-sample-dropout head.                        |

## Data pipeline

- **70 / 15 / 15 stratified train / val / test split.** The test split is held out from training, grid search, ensemble-weight search and threshold tuning, so the reported test F1 is unbiased.
- **Preprocessing** (`scripts/models.py::clean_text`): `HTTPURL` / `@USER` placeholders (matching BERTweet's pretraining), hashtag word kept but `#` dropped, elongation collapsed (`looool` → `lool`), HTML entities decoded, optional emoji demojize if the `emoji` package is installed.
- **Keyword integration**: when Kaggle's `keyword` column is present we prepend it as a natural-language topic prefix — `"flood. {tweet body}"`. URL-encoded keywords (`airplane%20accident`) are decoded first.
- **Dynamic padding** via `DataCollatorWithPadding` (pads per batch, not to a fixed `MAX_LEN`).

## Training

- **AdamW** with weight-decay grouping that excludes bias / LayerNorm.
- **Layer-wise learning-rate decay** for the transformers (deeper layers fine-tune fastest, embeddings slowest).
- **Linear warmup** (10% of steps) then linear decay.
- **Mixed precision** (`torch.cuda.amp`) on CUDA, gradient clipping at max-norm 1.0.
- **Class-imbalance handling**: BCE `pos_weight` for the LSTM, balanced CE class weights for the transformers.
- **Early stopping** on val F1 with patience 2; the best-by-val state is kept (not the final-epoch state).

## Hyperparameter grid search

Run `python scripts/models.py`. The script:

1. Trains every config in a small per-model grid and keeps the best by validation F1:
   - **LSTM**: `lr ∈ {5e-4, 1e-3}` × `hidden_dim ∈ {128, 256}` → 4 runs
   - **BERTweet** and **RoBERTa**: `base_lr ∈ {1e-5, 2e-5, 3e-5}` × `layer_decay ∈ {0.9, 0.95}` → 6 runs each
   - Total: **16 training runs**. Edit the `LSTM_GRID` / `TRANSFORMER_GRID` dicts near the bottom of `models.py` to expand or shrink.
2. Generates val / test / submission probabilities for each winning model.
3. Grid-searches over the ensemble weight simplex (step 0.05, 231 weight combos) and sweeps the decision threshold at each weight; picks the (weights, threshold) pair that maximizes **validation** F1.
4. Applies the val-tuned weights and threshold to the **held-out test split** for a single final F1.
5. Writes `export/grid_search_results.json` (full log of every trial) and `export/submission_ensemble.csv` (Kaggle submission), and auto-fills the results section of this README between the markers below.

## Results

### Split sizes

| Split                         | Rows | Positive rate |
| ----------------------------- | ---: | ------------: |
| Train                         | 5329 |         0.430 |
| Val                           | 1142 |         0.430 |
| Test                          | 1142 |         0.430 |
| Kaggle submission (unlabeled) | 3263 |            — |

### Best hyperparameters (selected by val F1)

| Model    | Best config                             | Val F1 |
| -------- | --------------------------------------- | -----: |
| LSTM     | `lr=0.001`, `hidden_dim=256`        | 0.7673 |
| BERTweet | `base_lr=3e-05`, `layer_decay=0.9`  | 0.8126 |
| RoBERTa  | `base_lr=1e-05`, `layer_decay=0.95` | 0.8130 |

### Best ensemble (selected by val F1)

| LSTM weight | BERTweet weight | RoBERTa weight | Threshold | Val F1 |
| ----------: | --------------: | -------------: | --------: | -----: |
|        0.40 |            0.20 |           0.40 |      0.54 | 0.8255 |

### Held-out test performance

| Model                                              |          Test F1 |         Test Acc |
| -------------------------------------------------- | ---------------: | ---------------: |
| LSTM                                               |           0.7564 |           0.7986 |
| BERTweet                                           |           0.8248 |           0.8538 |
| RoBERTa                                            |           0.8311 |           0.8590 |
| **Ensemble** (val-tuned weights & threshold) | **0.8189** | **0.8520** |

_The test split was held out from training, grid search, ensemble-weight search and threshold tuning._

### Performance curves (best config from each grid search)

| Train F1                                  | Val F1                                |
| ----------------------------------------- | ------------------------------------- |
| ![Train F1](plots/performance/train_f1.png) | ![Val F1](plots/performance/val_f1.png) |

| Train Loss                                    | Val Loss                                  |
| --------------------------------------------- | ----------------------------------------- |
| ![Train Loss](plots/performance/train_loss.png) | ![Val Loss](plots/performance/val_loss.png) |

| Train Accuracy                              | Val Accuracy                            |
| ------------------------------------------- | --------------------------------------- |
| ![Train Acc](plots/performance/train_acc.png) | ![Val Acc](plots/performance/val_acc.png) |

### Exploratory data analysis

| Label distribution                            | Word count by class                                   |
| --------------------------------------------- | ----------------------------------------------------- |
| ![Label pie](plots/eda/tweet_label_piChart.png) | ![Word count](plots/eda/tweet_word_count_histogram.png) |

| Mean word length                                           | Missing values                                |
| ---------------------------------------------------------- | --------------------------------------------- |
| ![Word length](plots/eda/tweet_word_count_histogram_pdf.png) | ![Missing values](plots/eda/missing_values.png) |

![Top keywords by class](plots/eda/top_keywords_by_class.png)

<!-- END AUTO-RESULTS -->
