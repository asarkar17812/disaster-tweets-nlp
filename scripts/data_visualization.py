"""
scripts/data_visualization.py
=============================

Exploratory data analysis (EDA) for the Kaggle disaster-tweets dataset.

What each plot tells you
------------------------
1.  **Class-balance pie chart** - confirms the ~57 / 43 split. The mild
    imbalance is why `models.py` uses `pos_weight` (BCE) and balanced
    class weights (CE) instead of uniform weighting.

2.  **Word-count histograms by class** - shows that disaster tweets tend
    to be slightly *longer* than non-disaster tweets (more nouns and
    place-names per message). It also confirms that `MAX_LEN=96` in
    `models.py` covers virtually all examples.

3.  **Mean word-length histograms by class** - disaster tweets contain
    more long words (place names, hashtags like `#earthquake`,
    organizations like `RescueTeam`). A subtle signal that the cased
    BERT tokenizer can pick up.

4.  **Missing-value summary** - `keyword` and `location` are sometimes
    missing. We do not currently feed these to the model; this plot
    documents what a future feature-engineering pass could exploit.

5.  **Top keywords by class** - the `keyword` column is a tagging field
    Kaggle provides; the most distinctive disaster-tagged keywords are a
    sanity check that the labels match intuition.

Engineering notes
-----------------
* Paths use `pathlib.Path` so the script works regardless of OS-level
  path separators.
* DPI dropped from 1200 to 200: the previous setting produced ~5 MB
  PNGs with no visible benefit at typical screen / report sizes.
* Typo fixed: "Pi Chart" -> "Pie Chart".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(r"F:\disaster_tweets")
SOURCE_DIR = PROJECT_ROOT / "source"
EDA_DIR = PROJECT_ROOT / "plots" / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# `skip_blank_lines=True` handles the leading blank lines in train.csv that
# would otherwise confuse strict CSV readers.
df_train = pd.read_csv(SOURCE_DIR / "train.csv", skip_blank_lines=True)

X = df_train["text"]
y = df_train["target"]

sns.set_theme(style="whitegrid")
DPI = 200


# ---------------------------------------------------------------------------
# 1. Class balance
# ---------------------------------------------------------------------------
# `(y == 1).sum()` is the idiomatic count of True rows; the previous
# `y[y == 1].count()` counts non-null rows of the filtered series, which is
# the same here but easier to misread.
n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
colors = ["#00FF2A", "#DF1616"]

plt.figure(figsize=(6, 4))
plt.pie(
    [n_pos, n_neg],
    labels=[f"Disaster ({n_pos})", f"Non-disaster ({n_neg})"],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
)
plt.title("Pie chart of training disaster tweets by label")
plt.tight_layout()
plt.savefig(EDA_DIR / "tweet_label_piChart.png", dpi=DPI, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 2. Word-count histograms by class
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
disaster_lens = df_train.loc[df_train["target"] == 1, "text"].str.split().map(len)
non_disaster_lens = df_train.loc[df_train["target"] == 0, "text"].str.split().map(len)
ax1.hist(disaster_lens, bins=30, color="#1f77b4")
ax1.set_title("Disaster tweets")
ax1.set_xlabel("words per tweet")
ax1.set_ylabel("frequency")
ax2.hist(non_disaster_lens, bins=30, color="#ff7f0e")
ax2.set_title("Non-disaster tweets")
ax2.set_xlabel("words per tweet")
fig.suptitle("Words per tweet (before tokenization)")
plt.tight_layout()
plt.savefig(
    EDA_DIR / "tweet_word_count_histogram.png", dpi=DPI, bbox_inches="tight"
)
plt.close()


# ---------------------------------------------------------------------------
# 3. Mean word-length distributions by class (KDE-smoothed)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
disaster_mean_wl = df_train.loc[df_train["target"] == 1, "text"].str.split().apply(
    lambda toks: float(np.mean([len(t) for t in toks])) if toks else 0.0
)
non_disaster_mean_wl = df_train.loc[df_train["target"] == 0, "text"].str.split().apply(
    lambda toks: float(np.mean([len(t) for t in toks])) if toks else 0.0
)
sns.histplot(disaster_mean_wl, kde=True, ax=ax1, color="purple")
ax1.set_title("Disaster tweets")
ax1.set_xlabel("mean word length (chars)")
sns.histplot(non_disaster_mean_wl, kde=True, ax=ax2, color="orange")
ax2.set_title("Non-disaster tweets")
ax2.set_xlabel("mean word length (chars)")
fig.suptitle("Mean word length per tweet")
plt.tight_layout()
plt.savefig(
    EDA_DIR / "tweet_word_count_histogram_pdf.png",
    dpi=DPI,
    bbox_inches="tight",
)
plt.close()


# ---------------------------------------------------------------------------
# 4. Missing-value summary (keyword / location)
# ---------------------------------------------------------------------------
missing = df_train.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df_train) * 100).round(2)
print("Missing values per column:")
for col, n in missing.items():
    print(f"  {col:10s} {n:5d}  ({missing_pct[col]}%)")

plt.figure(figsize=(7, 4))
sns.barplot(x=missing.values, y=missing.index, palette="rocket")
plt.title("Missing values per column (train)")
plt.xlabel("count of missing rows")
plt.tight_layout()
plt.savefig(EDA_DIR / "missing_values.png", dpi=DPI, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# 5. Top keywords by class
# ---------------------------------------------------------------------------
# `keyword` is a sparse tag (sometimes missing). For the rows where it is
# present, looking at the top keywords associated with each label is a
# quick sanity check that the labels behave intuitively.
top_n = 15
kw = df_train.dropna(subset=["keyword"]).copy()
top_disaster = (
    kw.loc[kw["target"] == 1, "keyword"].value_counts().head(top_n)
)
top_non = kw.loc[kw["target"] == 0, "keyword"].value_counts().head(top_n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(x=top_disaster.values, y=top_disaster.index, ax=ax1, palette="Reds_r")
ax1.set_title(f"Top {top_n} keywords - disaster tweets")
ax1.set_xlabel("count")
sns.barplot(x=top_non.values, y=top_non.index, ax=ax2, palette="Blues_r")
ax2.set_title(f"Top {top_n} keywords - non-disaster tweets")
ax2.set_xlabel("count")
plt.tight_layout()
plt.savefig(EDA_DIR / "top_keywords_by_class.png", dpi=DPI, bbox_inches="tight")
plt.close()


print(f"\nEDA plots written to {EDA_DIR}")