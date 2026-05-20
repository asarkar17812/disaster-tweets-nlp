"""
scripts/models.py
=================

Trains a **BERTweet + RoBERTa + improved-LSTM** soft-voting ensemble for
the Kaggle "Real or Not? NLP with Disaster Tweets" task with a
**grid search over per-model hyperparameters** *and* a search over
**ensemble soft-voting weights** + decision threshold.

Pipeline
--------
1. Preprocess + 70 / 15 / 15 stratified split (train / val / **held-out test**).
2. For each model (LSTM, BERTweet, RoBERTa) run a small grid search,
   picking the config with the best validation F1.
3. Using the three winning configs, score val / test / submission.
4. Search over ensemble weights on the probability simplex (and a
   threshold sweep at each weight) to maximize validation F1.
5. Apply the val-tuned weights & threshold to the **held-out test
   split** for a single, unbiased final F1.
6. Write a Kaggle submission CSV, a grid_search_results.json log, and
   auto-update the README "AUTO-RESULTS" section with the
   hyperparameters, metrics, and figure embeds.

Why this lineup
---------------
* **BERTweet (`vinai/bertweet-base`)** — A RoBERTa-architecture model
  pretrained on ~850M English tweets. Tweets are short, noisy and
  hashtag-heavy: BERTweet sees that distribution at scale during
  pretraining, so its representations transfer far better than
  general-domain BERT. Empirically this is the single biggest win on
  the disaster-tweets dataset. (Replaces `bert-base-cased`.)
* **RoBERTa (`roberta-base`)** — General-domain RoBERTa keeps ensemble
  diversity: trained on a totally different corpus (web + books) so
  its errors are only partially correlated with BERTweet's, and
  averaging probabilities reduces variance.
* **Improved LSTM** — kept for diversity. Versus the previous LSTM
  this version adds attention pooling (was a flat mean), multi-sample
  dropout in the head, and is sized via grid search.

Engineering choices that materially affect F1
---------------------------------------------
* **70 / 15 / 15 stratified split.** The labeled set is split into
  train / val / test with class-ratio preservation across all three;
  the test split is locked away until *after* hyperparameter search,
  ensemble-weight search, and threshold tuning, so the reported test
  F1 is an unbiased estimate.
* **Keyword-column integration.** Kaggle provides a `keyword` field
  (often missing). When present, it is prepended to the text as a
  natural-language topic prefix (`"flood. {tweet body}"`), giving
  the transformer an explicit topical cue to attend to. URL-encoded
  keywords (`airplane%20accident`) are decoded.
* **`HTTPURL` / `@USER` placeholders** in `clean_text` — match
  BERTweet's pretraining; harmless for the other models.
* **Character-elongation collapse** (`looool` -> `lool`) — reduces
  out-of-vocab subword fragments without erasing emphasis entirely.
* **Optional emoji demojize** — if the `emoji` package is installed,
  emojis become text tokens (e.g. `:fire:`), matching BERTweet's
  pretraining; otherwise the pipeline still runs.
* **Layer-wise learning-rate decay (LLRD)** for transformers — deeper
  layers fine-tune fastest, embeddings slowest; we search the decay
  factor along with the base LR.
* **Multi-sample dropout** in the LSTM head — averages logits across
  several dropout masks; cheap regularization worth ~0.1-0.3 F1.
* **Class-imbalance handling** — BCE `pos_weight` for the LSTM and
  balanced CE class weights for the transformers (the dataset is
  ~57/43, mild imbalance).
* **AMP / mixed precision** on CUDA, gradient clipping at max-norm 1.0,
  weight-decay grouping that excludes bias / LayerNorm, early stopping
  on val F1 with patience 2 — standard recipes carried over.
* **Probability-space soft voting.** Each model's logits are converted
  to probabilities (softmax for transformers, sigmoid for the LSTM)
  *before* averaging.
* **Reproducibility.** Fixed seeds for `random`, `numpy`, `torch`
  (CPU and CUDA). Grid configs are compared on identical splits.
"""

from __future__ import annotations

import html
import itertools
import json
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset as HFDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device} | AMP: {USE_AMP}")


def amp_context():
    """Mixed-precision context. No-op on CPU."""
    return torch.cuda.amp.autocast() if USE_AMP else nullcontext()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"F:\disaster_tweets")
SOURCE_DIR = PROJECT_ROOT / "source"
PLOTS_DIR = PROJECT_ROOT / "plots" / "performance"
EXPORT_DIR = PROJECT_ROOT / "export"
README_PATH = PROJECT_ROOT / "README.md"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
# train.csv has blank leading lines; `skip_blank_lines=True` handles it.
labeled_df = pd.read_csv(SOURCE_DIR / "train.csv", skip_blank_lines=True)
submission_df = pd.read_csv(SOURCE_DIR / "test.csv", skip_blank_lines=True)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WHITESPACE_RE = re.compile(r"\s+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")  # 3+ repeats -> 2

# Optional emoji demojization (matches BERTweet's pretraining).
# If the `emoji` package isn't installed, the pipeline still runs.
try:
    import emoji as _emoji_lib  # type: ignore

    def _emoji_to_text(s: str) -> str:
        return _emoji_lib.demojize(s, delimiters=(" :", ": "))
except ImportError:  # pragma: no cover
    def _emoji_to_text(s: str) -> str:
        return s


def clean_text(text: str) -> str:
    """Tweet-aware cleaning shared across all three models.

    * **`HTTPURL` / `@USER` placeholders** — the literal tokens
      BERTweet was pretrained on; harmless for the other models.
    * **Strip the `#`, keep the hashtag word.** `#earthquake` carries
      the topical signal; the `#` itself doesn't.
    * **Collapse elongation**: `looooove` -> `loove`. Keeps a doubled
      letter as a soft emphasis cue without producing OOV subwords.
    * **Casing and punctuation preserved** — cased models benefit
      from proper-noun and ALL-CAPS signal.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = URL_RE.sub("HTTPURL", text)
    text = MENTION_RE.sub("@USER", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = _emoji_to_text(text)
    text = ELONGATION_RE.sub(r"\1\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_keyword(kw) -> str:
    """Decode and normalize Kaggle's `keyword` column.

    `airplane%20accident` -> `airplane accident`. Returns `""` for
    missing values.
    """
    if not isinstance(kw, str):
        return ""
    return unquote(kw).strip().lower()


def build_input(text_clean: str, kw_clean: str) -> str:
    """`"<keyword>. <body>"` when keyword is present, else body only."""
    if kw_clean:
        return f"{kw_clean}. {text_clean}"
    return text_clean


for df in (labeled_df, submission_df):
    df["text_clean"] = df["text"].apply(clean_text)
    df["keyword_clean"] = df["keyword"].apply(clean_keyword)
    df["input_text"] = [
        build_input(t, k) for t, k in zip(df["text_clean"], df["keyword_clean"])
    ]

# Drop rows whose `input_text` is empty after cleaning — labeled only.
# The submission set keeps every row so the output CSV matches Kaggle's
# expected row count.
labeled_df = labeled_df[labeled_df["input_text"].str.len() > 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 70 / 15 / 15 stratified split
# ---------------------------------------------------------------------------
# Two-stage stratified split:
#   stage 1:   85% trainval  |  15% test
#   stage 2:   inside trainval: test_size = 0.15 / 0.85 = 0.17647...
#              -> val   = 0.85 * 0.17647 = 0.15 of original
#              -> train = 0.85 * 0.82353 = 0.70 of original
# Result: exactly 70 / 15 / 15.
trainval_df, test_split = train_test_split(
    labeled_df,
    test_size=0.15,
    random_state=SEED,
    stratify=labeled_df["target"],
)
train_split, val_split = train_test_split(
    trainval_df,
    test_size=0.15 / 0.85,
    random_state=SEED,
    stratify=trainval_df["target"],
)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)
test_split = test_split.reset_index(drop=True)

print(
    f"Train: {len(train_split)} ({train_split['target'].mean():.3f} pos) | "
    f"Val: {len(val_split)} ({val_split['target'].mean():.3f} pos) | "
    f"Test: {len(test_split)} ({test_split['target'].mean():.3f} pos) | "
    f"Kaggle submission: {len(submission_df)}"
)


# ---------------------------------------------------------------------------
# Transformer arm: BERTweet (tweet-pretrained) + RoBERTa (general)
# ---------------------------------------------------------------------------
MODEL_NAMES = {
    "BERTweet": "vinai/bertweet-base",
    "RoBERTa": "roberta-base",
}
MAX_LEN = 96  # ~99th percentile of WordPiece/BPE-tokenized tweet length.


def _load_tok(ckpt: str):
    """Pass `normalization=False` for BERTweet's slow tokenizer; fall
    back gracefully for tokenizers that don't accept the kwarg."""
    try:
        return AutoTokenizer.from_pretrained(ckpt, normalization=False)
    except TypeError:
        return AutoTokenizer.from_pretrained(ckpt)


tokenizers = {name: _load_tok(ckpt) for name, ckpt in MODEL_NAMES.items()}


def tokenize_fn(batch, tokenizer):
    # No padding here — `DataCollatorWithPadding` pads per batch.
    return tokenizer(batch["input_text"], truncation=True, max_length=MAX_LEN)


def _to_hf(df: pd.DataFrame, has_labels: bool) -> HFDataset:
    cols = ["input_text", "target"] if has_labels else ["input_text"]
    ds = HFDataset.from_pandas(df[cols], preserve_index=False)
    if has_labels:
        ds = ds.map(lambda ex: {"labels": int(ex["target"])})
    return ds


tokenized_datasets: dict[str, dict[str, HFDataset]] = {}
for name, tokenizer in tokenizers.items():
    train_tok = _to_hf(train_split, True).map(
        lambda b, t=tokenizer: tokenize_fn(b, t), batched=True
    )
    val_tok = _to_hf(val_split, True).map(
        lambda b, t=tokenizer: tokenize_fn(b, t), batched=True
    )
    test_tok = _to_hf(test_split, True).map(
        lambda b, t=tokenizer: tokenize_fn(b, t), batched=True
    )
    sub_tok = _to_hf(submission_df, False).map(
        lambda b, t=tokenizer: tokenize_fn(b, t), batched=True
    )

    keep_labeled = {"input_ids", "attention_mask", "labels"}
    keep_unlabeled = {"input_ids", "attention_mask"}
    if "token_type_ids" in train_tok.column_names:
        keep_labeled.add("token_type_ids")
        keep_unlabeled.add("token_type_ids")

    train_tok = train_tok.remove_columns(
        [c for c in train_tok.column_names if c not in keep_labeled]
    )
    val_tok = val_tok.remove_columns(
        [c for c in val_tok.column_names if c not in keep_labeled]
    )
    test_tok = test_tok.remove_columns(
        [c for c in test_tok.column_names if c not in keep_labeled]
    )
    sub_tok = sub_tok.remove_columns(
        [c for c in sub_tok.column_names if c not in keep_unlabeled]
    )
    tokenized_datasets[name] = {
        "train": train_tok,
        "val": val_tok,
        "test": test_tok,
        "sub": sub_tok,
    }


BATCH_SIZE = 32

data_collators = {
    name: DataCollatorWithPadding(tokenizer=tokenizers[name]) for name in MODEL_NAMES
}


def _loader(ds: HFDataset, name: str, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle,
        collate_fn=data_collators[name],
    )


train_loaders = {n: _loader(tokenized_datasets[n]["train"], n, True) for n in MODEL_NAMES}
val_loaders = {n: _loader(tokenized_datasets[n]["val"], n, False) for n in MODEL_NAMES}
test_loaders = {n: _loader(tokenized_datasets[n]["test"], n, False) for n in MODEL_NAMES}
sub_loaders = {n: _loader(tokenized_datasets[n]["sub"], n, False) for n in MODEL_NAMES}


# ---------------------------------------------------------------------------
# LSTM arm — uses BERT-cased tokenizer so the embedding-init transfer
# remains a direct vocab-to-vocab copy.
# ---------------------------------------------------------------------------
LSTM_INIT_CKPT = "bert-base-cased"
lstm_tokenizer = AutoTokenizer.from_pretrained(LSTM_INIT_CKPT)
PAD_ID = lstm_tokenizer.pad_token_id


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int = MAX_LEN):
        self.encodings = tokenizer(
            list(texts), truncation=True, max_length=max_len
        )
        self.labels = (
            torch.tensor(list(labels), dtype=torch.float) if labels is not None else None
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        ids = torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long)
        mask = torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long)
        label = self.labels[idx] if self.labels is not None else torch.tensor(0.0)
        return ids, mask, label


def lstm_collate(batch):
    ids, masks, labels = zip(*batch)
    ids = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=PAD_ID)
    masks = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return ids, masks, labels


def _lstm_loader(df, has_labels, shuffle):
    return DataLoader(
        TweetDataset(
            df["input_text"],
            df["target"] if has_labels else None,
            lstm_tokenizer,
        ),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=lstm_collate,
    )


train_lstm_loader = _lstm_loader(train_split, True, True)
val_lstm_loader = _lstm_loader(val_split, True, False)
test_lstm_loader = _lstm_loader(test_split, True, False)
sub_lstm_loader = _lstm_loader(submission_df, False, False)


# ---------------------------------------------------------------------------
# LSTM architecture
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """Learnable additive attention pooling over the time axis.

    Replaces flat mean-pool: the model learns *which* tokens drive
    the decision, so it can downweight filler ("just got home and...")
    in favor of topical cues ("...there's a fire").
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, hiddens, mask):
        logits = self.score(hiddens).squeeze(-1)            # (B, T)
        logits = logits.masked_fill(mask == 0, -1e4)        # fp16-safe
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)   # (B, T, 1)
        return (hiddens * weights).sum(dim=1)               # (B, H)


class MultiSampleDropout(nn.Module):
    """Average logits across multiple dropout masks before loss."""

    def __init__(self, num_samples: int = 5, p: float = 0.5):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])

    def forward(self, x, fc: nn.Module):
        return torch.stack([fc(d(x)) for d in self.dropouts], dim=0).mean(dim=0)


class LSTMClassifier(nn.Module):
    """BiLSTM + attention pool + multi-sample dropout head. Returns raw logits."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 768,
        hidden_dim: int = 128,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        ms_dropout: float = 0.5,
        ms_samples: int = 5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = AttentionPool(out_dim)
        self.fc = nn.Linear(out_dim, 1)
        self.head_dropout = MultiSampleDropout(num_samples=ms_samples, p=ms_dropout)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        pooled = self.attn(out, attention_mask)
        return self.head_dropout(pooled, self.fc).squeeze(-1)


# Cache BERT-cased's embedding matrix once: re-using it across grid-search
# trials avoids reloading bert-base-cased from disk for every LSTM run.
_bert_emb_cache: torch.Tensor | None = None


def get_bert_emb_cache() -> torch.Tensor:
    global _bert_emb_cache
    if _bert_emb_cache is None:
        bert = AutoModel.from_pretrained(LSTM_INIT_CKPT)
        _bert_emb_cache = bert.embeddings.word_embeddings.weight.detach().clone()
        del bert
        print(f"  Cached {LSTM_INIT_CKPT} word embeddings: {tuple(_bert_emb_cache.shape)}")
    return _bert_emb_cache


def init_lstm_embeddings(model: LSTMClassifier) -> None:
    """Copy BERT-cased's pretrained word-embedding matrix into the LSTM."""
    src = get_bert_emb_cache()
    dst = model.embedding.weight
    if src.shape == dst.shape:
        with torch.no_grad():
            dst.copy_(src)
    else:
        # Down-/up-project if grid picks a non-768 embed_dim.
        proj = nn.Linear(src.shape[1], dst.shape[1], bias=False)
        with torch.no_grad():
            dst.copy_(proj(src))


# ---------------------------------------------------------------------------
# Class weighting (~57 / 43 imbalance)
# ---------------------------------------------------------------------------
pos_count = int(train_split["target"].sum())
neg_count = len(train_split) - pos_count
pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
class_weights_np = compute_class_weight(
    "balanced", classes=np.array([0, 1]), y=train_split["target"].values
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float, device=device)
print(
    f"pos_weight={pos_weight.item():.3f} | "
    f"class_weights={class_weights.cpu().numpy().round(3).tolist()}"
)


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------
NO_DECAY = ("bias", "LayerNorm.weight")


def get_param_groups(model: nn.Module, weight_decay: float = 0.01):
    """No weight decay on bias / LayerNorm — standard recipe."""
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if any(nd in n for nd in NO_DECAY) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def llrd_param_groups(
    model: nn.Module,
    base_lr: float = 2e-5,
    layer_decay: float = 0.95,
    weight_decay: float = 0.01,
):
    """Layer-wise learning-rate decay (LLRD).

    Higher transformer layers get the full `base_lr`; lower layers
    and embeddings get `base_lr * layer_decay ** depth_from_top`.
    Passing `layer_decay=1.0` recovers uniform LR.
    """
    pnames = [n for n, _ in model.named_parameters()]
    if any(n.startswith("bert.encoder.layer.") for n in pnames):
        enc_prefix, emb_prefix = "bert.encoder.layer.", "bert.embeddings"
    elif any(n.startswith("roberta.encoder.layer.") for n in pnames):
        enc_prefix, emb_prefix = "roberta.encoder.layer.", "roberta.embeddings"
    else:
        return get_param_groups(model, weight_decay)

    n_layers = max(
        int(n.split(".")[3]) for n in pnames if n.startswith(enc_prefix)
    ) + 1

    def lr_for(name: str) -> float:
        if name.startswith(enc_prefix):
            layer_i = int(name.split(".")[3])
            return base_lr * (layer_decay ** (n_layers - 1 - layer_i))
        if name.startswith(emb_prefix):
            return base_lr * (layer_decay ** n_layers)
        return base_lr

    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = lr_for(n)
        wd = 0.0 if any(nd in n for nd in NO_DECAY) else weight_decay
        groups.setdefault((lr, wd), []).append(p)
    return [
        {"params": ps, "lr": lr, "weight_decay": wd}
        for (lr, wd), ps in groups.items()
    ]


def best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds in [0.20, 0.80] (step 0.01); return (t, f1)."""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.01):
        f = f1_score(labels, (probs >= t).astype(int))
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


# ---------------------------------------------------------------------------
# Training / evaluation primitives
# ---------------------------------------------------------------------------
def _step_amp(loss, optimizer, scaler, params):
    """Backward + grad-clip + step, with or without AMP."""
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()


def train_lstm_epoch(model, loader, loss_fn, optimizer, scaler):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
        _step_amp(loss, optimizer, scaler, model.parameters())
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits.detach().float()).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    preds = (np.array(all_probs) >= 0.5).astype(int)
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, preds),
        f1_score(all_labels, preds),
    )


@torch.no_grad()
def eval_lstm(model, loader, loss_fn=None):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    n_batches = 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        with amp_context():
            logits = model(ids, mask)
            if loss_fn is not None:
                total_loss += loss_fn(logits, labels).item()
        all_probs.extend(torch.sigmoid(logits.float()).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        n_batches += 1
    probs = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)
    avg_loss = total_loss / n_batches if loss_fn is not None else float("nan")
    return (
        avg_loss,
        accuracy_score(labels_arr, preds),
        f1_score(labels_arr, preds),
        probs,
        labels_arr,
    )


@torch.no_grad()
def predict_lstm(model, loader) -> np.ndarray:
    model.eval()
    out = []
    for ids, mask, _ in loader:
        ids, mask = ids.to(device), mask.to(device)
        with amp_context():
            logits = model(ids, mask)
        out.extend(torch.sigmoid(logits.float()).cpu().numpy())
    return np.array(out)


def train_transformer_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        _step_amp(loss, optimizer, scaler, model.parameters())
        scheduler.step()
        total_loss += loss.item()
        probs = F.softmax(logits.detach().float(), dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    preds = (np.array(all_probs) >= 0.5).astype(int)
    return (
        total_loss / len(loader),
        accuracy_score(all_labels, preds),
        f1_score(all_labels, preds),
    )


@torch.no_grad()
def eval_transformer(model, loader, with_loss: bool = True):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        with amp_context():
            logits = model(**inputs).logits
            if with_loss:
                total_loss += F.cross_entropy(
                    logits, labels, weight=class_weights
                ).item()
        probs = F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        n_batches += 1
    probs = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)
    avg_loss = total_loss / n_batches if with_loss else float("nan")
    return (
        avg_loss,
        accuracy_score(labels_arr, preds),
        f1_score(labels_arr, preds),
        probs,
        labels_arr,
    )


@torch.no_grad()
def predict_transformer(model, loader) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with amp_context():
            logits = model(**batch).logits
        out.extend(F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.array(out)


# ---------------------------------------------------------------------------
# Per-config trainers (the unit of work for the grid search)
# ---------------------------------------------------------------------------
EPOCHS_LSTM = 8
EPOCHS_TRANSFORMER = 4
PATIENCE = 2


def _empty_run_log() -> dict:
    return {k: [] for k in
            ("train_loss", "train_acc", "train_f1",
             "val_loss", "val_acc", "val_f1")}


def train_one_lstm(
    config: dict,
    train_loader,
    val_loader,
    epochs: int = EPOCHS_LSTM,
    patience: int = PATIENCE,
):
    """Train a single LSTM config. Returns (best_val_f1, best_state, run_log)."""
    model = LSTMClassifier(
        vocab_size=lstm_tokenizer.vocab_size,
        pad_id=PAD_ID,
        embed_dim=config.get("embed_dim", 768),
        hidden_dim=config["hidden_dim"],
        n_layers=config.get("n_layers", 2),
        dropout=config.get("dropout", 0.3),
        ms_dropout=config.get("ms_dropout", 0.5),
    ).to(device)
    init_lstm_embeddings(model)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(get_param_groups(model, 0.01), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_f1 = -1.0
    best_state = None
    run_log = _empty_run_log()
    stale = 0
    for epoch in range(epochs):
        tr_loss, tr_acc, tr_f1 = train_lstm_epoch(
            model, train_loader, loss_fn, optimizer, scaler
        )
        va_loss, va_acc, va_f1, _, _ = eval_lstm(model, val_loader, loss_fn)
        run_log["train_loss"].append(tr_loss)
        run_log["train_acc"].append(tr_acc)
        run_log["train_f1"].append(tr_f1)
        run_log["val_loss"].append(va_loss)
        run_log["val_acc"].append(va_acc)
        run_log["val_f1"].append(va_f1)
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_f1, best_state, run_log


def train_one_transformer(
    name: str,
    ckpt: str,
    config: dict,
    train_loader,
    val_loader,
    epochs: int = EPOCHS_TRANSFORMER,
    patience: int = PATIENCE,
):
    """Train one transformer config. Returns (best_val_f1, best_state, run_log)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=2
    ).to(device)

    optimizer = AdamW(
        llrd_param_groups(
            model,
            base_lr=config["base_lr"],
            layer_decay=config.get("layer_decay", 0.95),
            weight_decay=config.get("weight_decay", 0.01),
        )
    )
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(config.get("warmup_ratio", 0.1) * num_training_steps)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_f1 = -1.0
    best_state = None
    run_log = _empty_run_log()
    stale = 0
    for epoch in range(epochs):
        tr_loss, tr_acc, tr_f1 = train_transformer_epoch(
            model, train_loader, optimizer, scheduler, scaler
        )
        va_loss, va_acc, va_f1, _, _ = eval_transformer(model, val_loader)
        run_log["train_loss"].append(tr_loss)
        run_log["train_acc"].append(tr_acc)
        run_log["train_f1"].append(tr_f1)
        run_log["val_loss"].append(va_loss)
        run_log["val_acc"].append(va_acc)
        run_log["val_f1"].append(va_f1)
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_f1, best_state, run_log


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def _expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of the values in `grid`. Returns list of dicts."""
    keys = list(grid.keys())
    out = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        out.append(dict(zip(keys, combo)))
    return out


def grid_search(
    name: str,
    grid: dict[str, list],
    train_fn,
) -> dict:
    """Run a grid search; return the winning config + state + log + trials."""
    configs = _expand_grid(grid)
    print(f"\n=== Grid search: {name} ({len(configs)} configs) ===")
    best = {"val_f1": -1.0, "config": None, "state": None, "run_log": None}
    trials = []
    for i, config in enumerate(configs):
        t0 = time.time()
        print(f"  [{i+1}/{len(configs)}] {name} config: {config}")
        val_f1, state, run_log = train_fn(config)
        elapsed = time.time() - t0
        trials.append({"config": config, "val_f1": float(val_f1),
                       "elapsed_sec": round(elapsed, 1)})
        print(f"     -> val F1 {val_f1:.4f}   ({elapsed:.1f}s)")
        if val_f1 > best["val_f1"]:
            best = {"val_f1": float(val_f1), "config": config,
                    "state": state, "run_log": run_log}
    print(f"  Best {name}: F1 {best['val_f1']:.4f}  config={best['config']}")
    return {"best": best, "trials": trials}


# ---------------------------------------------------------------------------
# Ensemble weight search (simplex grid + threshold sweep)
# ---------------------------------------------------------------------------
def search_ensemble_weights(
    val_probs_per_model: dict[str, np.ndarray],
    val_labels: np.ndarray,
    step: float = 0.05,
) -> dict:
    """Grid over weights {w >= 0, sum(w) = 1} with the given step.

    For each weight combo we additionally do a threshold sweep in
    [0.20, 0.80]; the (weights, threshold) pair that maximizes val
    F1 is returned. With 3 models and step=0.05 this is 231 weight
    combos * 60 thresholds * one vectorized F1 = sub-second.
    """
    names = list(val_probs_per_model.keys())
    n = len(names)
    k = int(round(1.0 / step))

    best = {"weights": None, "threshold": None, "val_f1": -1.0}
    for combo in itertools.product(range(k + 1), repeat=n):
        if sum(combo) != k:
            continue
        ws = np.array(combo) / k
        weighted = np.zeros_like(val_labels, dtype=float)
        for w, name in zip(ws, names):
            weighted = weighted + w * val_probs_per_model[name]
        t, f = best_threshold(weighted, val_labels)
        if f > best["val_f1"]:
            best = {
                "weights": {name: float(w) for name, w in zip(names, ws)},
                "threshold": float(t),
                "val_f1": float(f),
            }
    return best


def weighted_average(
    probs_per_model: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    return sum(weights[n] * probs_per_model[n] for n in probs_per_model)


# ---------------------------------------------------------------------------
# Grid-search execution
# ---------------------------------------------------------------------------
# These constants control how exhaustive the search is. Reduce the
# lists for a faster run, expand them for a more thorough one. The
# defaults strike a balance: 4 LSTM + 6 BERTweet + 6 RoBERTa = 16
# training runs total.
LSTM_GRID = {
    "lr": [5e-4, 1e-3],
    "hidden_dim": [128, 256],
}
TRANSFORMER_GRID = {
    "base_lr": [1e-5, 2e-5, 3e-5],
    "layer_decay": [0.9, 0.95],
}
ENSEMBLE_WEIGHT_STEP = 0.05  # 0.05 -> 231 weight combos to evaluate (fast)


run_start = time.time()

# --- LSTM grid -----------------------------------------------------------
lstm_search = grid_search(
    "LSTM",
    LSTM_GRID,
    train_fn=lambda cfg: train_one_lstm(cfg, train_lstm_loader, val_lstm_loader),
)

# --- Transformer grids ---------------------------------------------------
transformer_searches: dict[str, dict] = {}
for name, ckpt in MODEL_NAMES.items():
    transformer_searches[name] = grid_search(
        name,
        TRANSFORMER_GRID,
        train_fn=lambda cfg, n=name, c=ckpt: train_one_transformer(
            n, c, cfg, train_loaders[n], val_loaders[n]
        ),
    )

elapsed_search = time.time() - run_start
print(f"\n=== Grid search done in {elapsed_search/60:.1f} min ===")


# ---------------------------------------------------------------------------
# Rebuild best models and collect val / test / submission probabilities
# ---------------------------------------------------------------------------
print("\n=== Rebuilding best models and collecting probabilities ===")

# Best LSTM
best_lstm_cfg = lstm_search["best"]["config"]
lstm_model = LSTMClassifier(
    vocab_size=lstm_tokenizer.vocab_size,
    pad_id=PAD_ID,
    embed_dim=best_lstm_cfg.get("embed_dim", 768),
    hidden_dim=best_lstm_cfg["hidden_dim"],
    n_layers=best_lstm_cfg.get("n_layers", 2),
    dropout=best_lstm_cfg.get("dropout", 0.3),
    ms_dropout=best_lstm_cfg.get("ms_dropout", 0.5),
).to(device)
lstm_model.load_state_dict(lstm_search["best"]["state"])

val_probs_per_model: dict[str, np.ndarray] = {}
test_probs_per_model: dict[str, np.ndarray] = {}
sub_probs_per_model: dict[str, np.ndarray] = {}

_, _, _, val_probs_per_model["LSTM"], val_labels = eval_lstm(lstm_model, val_lstm_loader)
_, _, _, test_probs_per_model["LSTM"], test_labels = eval_lstm(lstm_model, test_lstm_loader)
sub_probs_per_model["LSTM"] = predict_lstm(lstm_model, sub_lstm_loader)

# Best transformer instances
transformer_models: dict[str, nn.Module] = {}
for name, ckpt in MODEL_NAMES.items():
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=2
    ).to(device)
    model.load_state_dict(transformer_searches[name]["best"]["state"])
    transformer_models[name] = model

    _, _, _, val_probs_per_model[name], val_labels_t = eval_transformer(
        model, val_loaders[name]
    )
    _, _, _, test_probs_per_model[name], test_labels_t = eval_transformer(
        model, test_loaders[name]
    )
    sub_probs_per_model[name] = predict_transformer(model, sub_loaders[name])
    # All val/test loaders use shuffle=False over the same underlying split,
    # so label order must match across models. Assert to catch future drift.
    assert np.array_equal(val_labels, val_labels_t), f"val label order mismatch ({name})"
    assert np.array_equal(test_labels, test_labels_t), f"test label order mismatch ({name})"

model_order = ["LSTM"] + list(MODEL_NAMES)


# ---------------------------------------------------------------------------
# Per-model thresholds (diagnostics) and ensemble weight search
# ---------------------------------------------------------------------------
print("\n=== Per-model best thresholds on val (diagnostics) ===")
per_model_diag = {}
for name in model_order:
    t, f = best_threshold(val_probs_per_model[name], val_labels)
    per_model_diag[name] = {"threshold": float(t), "val_f1": float(f)}
    print(f"  {name:9s} threshold {t:.2f} -> val F1 {f:.4f}")

print(
    f"\n=== Ensemble weight search "
    f"(simplex grid step={ENSEMBLE_WEIGHT_STEP}) ==="
)
ens_search = search_ensemble_weights(
    val_probs_per_model, val_labels, step=ENSEMBLE_WEIGHT_STEP
)
print(
    f"  Best weights: "
    + ", ".join(f"{n}={w:.2f}" for n, w in ens_search["weights"].items())
)
print(
    f"  Best threshold: {ens_search['threshold']:.2f}  "
    f"-> val F1 {ens_search['val_f1']:.4f}"
)


# ---------------------------------------------------------------------------
# Held-out test set evaluation
# ---------------------------------------------------------------------------
val_ensemble = weighted_average(val_probs_per_model, ens_search["weights"])
test_ensemble = weighted_average(test_probs_per_model, ens_search["weights"])
sub_ensemble = weighted_average(sub_probs_per_model, ens_search["weights"])

t_star = ens_search["threshold"]
test_preds = (test_ensemble >= t_star).astype(int)
f_test_ens = float(f1_score(test_labels, test_preds))
acc_test_ens = float(accuracy_score(test_labels, test_preds))

print("\n=== Per-model held-out test metrics (threshold=0.5) ===")
test_per_model = {}
for name in model_order:
    preds = (test_probs_per_model[name] >= 0.5).astype(int)
    test_per_model[name] = {
        "test_f1": float(f1_score(test_labels, preds)),
        "test_acc": float(accuracy_score(test_labels, preds)),
    }
    print(
        f"  {name:9s} test F1 {test_per_model[name]['test_f1']:.4f}  "
        f"test acc {test_per_model[name]['test_acc']:.4f}"
    )

print(
    f"\n=== Held-out test (ENSEMBLE, val-tuned weights & threshold) ===\n"
    f"  F1   {f_test_ens:.4f}\n"
    f"  Acc  {acc_test_ens:.4f}\n"
    f"  Threshold {t_star:.2f}"
)


# ---------------------------------------------------------------------------
# Kaggle submission CSV
# ---------------------------------------------------------------------------
submission_preds = (sub_ensemble >= t_star).astype(int)
submission_out = pd.DataFrame(
    {"id": submission_df["id"].values, "target": submission_preds}
)
submission_path = EXPORT_DIR / "submission_ensemble.csv"
submission_out.to_csv(submission_path, index=False)
print(
    f"\nWrote Kaggle submission -> {submission_path}  "
    f"({int(submission_preds.sum())} positive / {len(submission_preds)} total)"
)


# ---------------------------------------------------------------------------
# Plot performance curves (best run from each grid search)
# ---------------------------------------------------------------------------
plot_data = {
    "LSTM": lstm_search["best"]["run_log"],
    "BERTweet": transformer_searches["BERTweet"]["best"]["run_log"],
    "RoBERTa": transformer_searches["RoBERTa"]["best"]["run_log"],
    # Ensemble has no per-epoch story: one val F1 from weight search.
    "Ensemble": {"val_f1": [ens_search["val_f1"]],
                 "val_acc": [float(accuracy_score(
                     val_labels, (val_ensemble >= t_star).astype(int)))]},
}


def plot_metrics(plot_data: dict, out_dir: Path) -> None:
    """One PNG per metric, one line per model (skipping empty)."""
    metrics = ["train_loss", "train_acc", "train_f1",
               "val_loss", "val_acc", "val_f1"]
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plotted = False
        for model_name, m in plot_data.items():
            if m.get(metric):
                plt.plot(range(1, len(m[metric]) + 1), m[metric],
                         marker="o", label=model_name)
                plotted = True
        if not plotted:
            plt.close()
            continue
        plt.title(metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=200, bbox_inches="tight")
        plt.close()


plot_metrics(plot_data, PLOTS_DIR)
print(f"Plots written to {PLOTS_DIR}")


# ---------------------------------------------------------------------------
# Persist a full results JSON
# ---------------------------------------------------------------------------
results_payload = {
    "split": {
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split),
        "kaggle_submission": len(submission_df),
        "train_pos_rate": float(train_split["target"].mean()),
        "val_pos_rate": float(val_split["target"].mean()),
        "test_pos_rate": float(test_split["target"].mean()),
    },
    "grids": {
        "LSTM": LSTM_GRID,
        "Transformer": TRANSFORMER_GRID,
        "ensemble_weight_step": ENSEMBLE_WEIGHT_STEP,
    },
    "best_configs": {
        "LSTM": lstm_search["best"]["config"],
        "BERTweet": transformer_searches["BERTweet"]["best"]["config"],
        "RoBERTa": transformer_searches["RoBERTa"]["best"]["config"],
    },
    "best_val_f1": {
        "LSTM": lstm_search["best"]["val_f1"],
        "BERTweet": transformer_searches["BERTweet"]["best"]["val_f1"],
        "RoBERTa": transformer_searches["RoBERTa"]["best"]["val_f1"],
    },
    "all_trials": {
        "LSTM": lstm_search["trials"],
        "BERTweet": transformer_searches["BERTweet"]["trials"],
        "RoBERTa": transformer_searches["RoBERTa"]["trials"],
    },
    "ensemble": {
        "weights": ens_search["weights"],
        "threshold": ens_search["threshold"],
        "val_f1": ens_search["val_f1"],
        "test_f1": f_test_ens,
        "test_acc": acc_test_ens,
    },
    "per_model_test": test_per_model,
    "per_model_val_thresholds": per_model_diag,
    "elapsed_grid_search_min": round(elapsed_search / 60, 2),
}
results_path = EXPORT_DIR / "grid_search_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=2)
print(f"Results JSON -> {results_path}")


# ---------------------------------------------------------------------------
# README auto-update
# ---------------------------------------------------------------------------
README_BEGIN = "<!-- BEGIN AUTO-RESULTS -->"
README_END = "<!-- END AUTO-RESULTS -->"


def _fmt_config(cfg: dict) -> str:
    return ", ".join(f"`{k}={v}`" for k, v in cfg.items())


def build_results_markdown(payload: dict) -> str:
    """Render the compact auto-generated block that lives between the
    README markers.

    The curated README above the markers covers narrative, figures and
    interpretation; this block is intentionally just the latest numeric
    tables, so re-runs don't churn the hand-written content.
    """
    best_cfg = payload["best_configs"]
    best_val = payload["best_val_f1"]
    ens = payload["ensemble"]
    test_pm = payload["per_model_test"]
    w = ens["weights"]

    lines: list[str] = ["", "### Auto-generated quick reference", ""]
    lines.append(
        "_Latest run of `scripts/models.py` "
        f"(grid search took {payload['elapsed_grid_search_min']} min). "
        "Full per-trial log in `export/grid_search_results.json`._"
    )
    lines.append("")

    lines.append("**Best per-model configs (selected by val F1)**")
    lines.append("")
    lines.append("| Model    | Best config                            | Val F1 |")
    lines.append("| -------- | -------------------------------------- | ------:|")
    for name in ["LSTM", "BERTweet", "RoBERTa"]:
        lines.append(
            f"| {name:<8s} | {_fmt_config(best_cfg[name])} | "
            f"{best_val[name]:.4f} |"
        )
    lines.append("")

    lines.append("**Best ensemble weights + threshold (selected by val F1)**")
    lines.append("")
    lines.append("| w(LSTM) | w(BERTweet) | w(RoBERTa) | Threshold | Val F1 |")
    lines.append("| ------: | ----------: | ---------: | --------: | -----: |")
    lines.append(
        f"| {w['LSTM']:.2f} | {w['BERTweet']:.2f} | {w['RoBERTa']:.2f} | "
        f"{ens['threshold']:.2f} | {ens['val_f1']:.4f} |"
    )
    lines.append("")

    lines.append("**Held-out test performance**")
    lines.append("")
    lines.append("| Model        | Test F1 | Test Acc |")
    lines.append("| ------------ | ------: | -------: |")
    for name in ["LSTM", "BERTweet", "RoBERTa"]:
        lines.append(
            f"| {name:<12s} | {test_pm[name]['test_f1']:.4f} | "
            f"{test_pm[name]['test_acc']:.4f} |"
        )
    lines.append(
        f"| **Ensemble** | **{ens['test_f1']:.4f}** | **{ens['test_acc']:.4f}** |"
    )
    lines.append("")

    return "\n".join(lines)


def update_readme(readme_path: Path, results_md: str) -> None:
    """Replace the content between `BEGIN AUTO-RESULTS` and `END AUTO-RESULTS`.

    If the markers are missing the function appends a new block to the
    end of the README and prints a warning rather than failing — that
    way a hand-written README without markers still gets the results.
    """
    if not readme_path.exists():
        readme_path.write_text(
            f"# Disaster Tweets\n\n{README_BEGIN}\n{results_md}\n{README_END}\n",
            encoding="utf-8",
        )
        print(f"README created -> {readme_path}")
        return

    text = readme_path.read_text(encoding="utf-8")
    if README_BEGIN in text and README_END in text:
        head, rest = text.split(README_BEGIN, 1)
        _, tail = rest.split(README_END, 1)
        new_text = f"{head}{README_BEGIN}\n{results_md}\n{README_END}{tail}"
        readme_path.write_text(new_text, encoding="utf-8")
        print(f"README updated -> {readme_path}")
    else:
        print(
            f"WARNING: {readme_path} has no AUTO-RESULTS markers; "
            "appending the results block at the end."
        )
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{README_BEGIN}\n{results_md}\n{README_END}\n")


update_readme(README_PATH, build_results_markdown(results_payload))

print(
    f"\nFINAL: held-out test F1 = {f_test_ens:.4f} "
    f"(val-tuned weights={ens_search['weights']}, threshold={t_star:.2f})"
)
