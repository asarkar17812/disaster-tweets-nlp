# Kaggle Disaster Tweets Competition:

This repo contains a project that I started over a year ago and have now decided to revisit it because I am now taking an Intro to Machine Learning class. I originally, and maybe naively, chose to use a soft-voting ensemble model that combines a Long-Short Term Memory Model (LSTM) with the popular BERT and RoBERTa transformers. As I continue to learn more, I plan to eventually redo this project and to improve the model architecture so that I may potentially use a hard-voting or stacking ensemble model and to remove the LSTM entirely. For now, I am marking this as a WIP.

## Model Performance by Epoch:

* The performance metrics listed below are the model's F1-score (closer to 1 is better) on the training set vs the validation set:

| EPOCH | LSTM             | BERT             | RoBERTa          |
| ----- | ---------------- | ---------------- | ---------------- |
| 1     | 0.0288 vs 0.0271 | 0.7493 vs 0.7819 | 0.7386 vs 0.7788 |
| 2     | 0.5394 vs 0.6417 | 0.8460 vs 0.7704 | 0.8311 vs 0.7990 |
| 3     | 0.6546 vs 0.6544 | 0.9061 vs 0.7692 | 0.8747 vs 0.7730 |
| 4     | 0.6861 vs 0.6740 | 0.9427 vs 0.7686 | 0.9054 vs 0.7758 |
| 5     | 0.7188 vs 0.6755 | 0.9620 vs 0.7719 | 0.9341 vs 0.7758 |
