"""
mental_health_EDA.py
====================
Exploratory Data Analysis (EDA) pipeline for a French mental-health social-media dataset.
Every graph is saved as a separate PNG file.

GRAPH FILES PRODUCED
--------------------
  01a_label_distribution_bar.png
  01b_label_distribution_pie.png
  02a_textlength_histogram_by_label.png
  02b_textlength_boxplot_by_label.png
  02c_charcount_boxplot_by_label.png
  03a_punctuation_normalized_bar.png
  03b_punctuation_normalized_table.png
  04_wordclouds_per_label.png
  05_cooccurrence_<label>.png
  06a_common_words_per_label.png
  06b_bigrams_per_label.png
  06c_trigrams_per_label.png
  07a_category_distribution_bar.png
  07b_category_distribution_pie.png
  08_category_label_heatmap.png
  10a_emoji_count_histogram.png
  10b_emoticon_count_histogram.png
  10c_emoji_presence_pie.png
  10d_avg_emoji_by_label.png
  10e_avg_emoticon_by_label.png
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import string
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import spacy
import emoji
from sklearn.feature_extraction.text import CountVectorizer

from typing import List, Tuple, Set, Dict, Optional
from nltk.tokenize import sent_tokenize

import nltk
try:
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet',   quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

warnings.filterwarnings("ignore")
print("All imports OK")


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Label colour map ──────────────────────────────────────────────────────────
LABEL_COLORS = {
    "Healthy":   "#D8BE3D",
    "Unhealthy": "#463EBD",
}

def label_palette(labels):
    """Return a list of colours aligned to the given label sequence."""
    return [LABEL_COLORS.get(str(lbl), "#888888") for lbl in labels]


class Config:
    CSV_PATH       = r"C:\Users\Admin\Documents\FYP\french dataset\Dataset\french_data.csv"
    OUTPUT_DIR     = r"MyResults"

    LANGUAGE_COL   = "language"
    LANGUAGE_VALUE = "French"
    TEXT_COL       = "text"
    LABEL_COL      = "mental_state"

    BG      = "white"        # ← all backgrounds white
    DPI     = 150
    PALETTE = "Set2"         # fallback for non-label plots

    TOP_N_WORDS = 20
    TOP_N_COOC  = 20


cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(f"Config ready — output folder: '{cfg.OUTPUT_DIR}'")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLOT HELPER
# ══════════════════════════════════════════════════════════════════════════════
class PlotHelper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        plt.rcParams.update({
            "figure.facecolor" : "white",   # ← white
            "axes.facecolor"   : "white",   # ← white
            "axes.spines.top"  : False,
            "axes.spines.right": False,
            "font.size"        : 11,
        })

    def save(self, filename: str) -> str:
        path = os.path.join(self.cfg.OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=self.cfg.DPI, bbox_inches="tight",
                    facecolor="white")          # ← force white on save
        plt.close()
        print(f"  [SAVED] {filename}")
        return path

    @staticmethod
    def safe_name(text: str) -> str:
        return re.sub(r'[\\/*?"<>|]+', "_", str(text)).strip()


helper = PlotHelper(cfg)
print("PlotHelper ready")


# ══════════════════════════════════════════════════════════════════════════════
# STOPWORD DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

PRONOUNS = {
    "je", "j", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "me", "moi", "te", "toi", "se",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs"
}

EXTRA_REMOVE = {
    "le", "la", "les", "un", "une", "des", "du", "au", "aux",
    "de", "à", "en", "dans", "sur", "avec", "pour", "par", "sans", "chez",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "que", "qui", "quand", "lorsque", "comme", "puisque",
    "quoique", "quoi", "si", "afin", "bien", "pendant",
    "avant", "après", "depuis", "jusqu", "malgré",
    "chaque", "tous", "toutes", "tout", "toute",
    "ce", "cet", "cette", "ces",
    "toujours", "parfois","autre","autres", "même", "mêmes",
    "cela", "ceci",
    "celui", "celle", "ceux", "celles",
    "celui-ci", "celui-là",
    "celle-ci", "celle-là",
    "ceux-ci", "ceux-là",
    "celles-ci", "celles-là",
}

NOISE = {"j", "m", "n", "s", "t", "quelqu", "aujourd", "hui", "pa"}

REMOVE_VERBS = {
    "être", "avoir",
    "suis", "es", "est", "sommes", "êtes", "sont",
    "étais", "était", "étions", "étiez", "étaient",
    "serai", "seras", "sera", "serons", "serez", "seront",
    "serais", "serait", "serions", "seriez", "seraient",
    "sois", "soit", "soyons", "soyez", "soient",
    "ai", "as", "a", "avons", "avez", "ont",
    "avais", "avait", "avions", "aviez", "avaient",
    "aurai", "auras", "aura", "aurons", "aurez", "auront",
    "aurais", "aurait", "aurions", "auriez", "auraient",
    "aie", "aies", "ait", "ayons", "ayez", "aient",
    "été", "eu",
}

KEEP_WORDS = {
    "ne", "pas", "rien", "personne", "jamais",
    "plus", "toujours", "parfois", "tellement", "trop",
    "dépression", "pensées", "vide", "douleur", "désespoir",
    "espoir", "suicidaires", "lumière", "obscurité", "âme",
    "résilience", "guérison",
}

STOPWORDS = (PRONOUNS | EXTRA_REMOVE | NOISE | REMOVE_VERBS) - KEEP_WORDS

WORDCLOUD_STOPWORDS = STOPWORDS | {
    "ne", "pas", "rien", "personne", "jamais",
    "très", "trop", "toujours", "parfois", "tellement",
    "plus", "bien", "vraiment", "encore", "déjà",
    "assez", "peu", "beaucoup", "moins", "autant",
}

nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT CLEANER
# ══════════════════════════════════════════════════════════════════════════════
class TextCleaner:
    def __init__(self, cfg: Config) -> None:
        self.cfg           = cfg
        self.nlp           = nlp
        self.stopwords_set = STOPWORDS

        self.emoji_regex = (
            r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|'
            r'[\U0001F680-\U0001F6FF]|[\U00002600-\U000026FF]|'
            r'[\U00002700-\U000027BF]|[\U0001F900-\U0001F9FF]|'
            r'[\U0001FA00-\U0001FA6F]|[\U0001FA70-\U0001FAFF]'
        )

        self.emoticon_patterns: List[str] = [
            r':\)|:-\)|:\]|=\]|=\)',
            r':\(|:-\(|:\[|=\[|=\(',
            r':D|:-D|=D',
            r';\)|;-\)',
            r':P|:-P|=P',
            r':o|:-o|:O|:-O',
            r':/|:-/',
            r":'\(",
            r'<3',
        ]

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.CSV_PATH, encoding="utf-8-sig")
        print(f"[TextCleaner] Total rows loaded  : {len(df)}")
        mask = (
            df[self.cfg.LANGUAGE_COL].str.strip().str.lower()
            == self.cfg.LANGUAGE_VALUE.lower()
        )
        df = df[mask].copy()
        before = len(df)
        df = df.drop_duplicates(subset=self.cfg.TEXT_COL).reset_index(drop=True)
        print(f"[TextCleaner] French rows kept   : {before}")
        print(f"[TextCleaner] After dedup        : {len(df)} ({before - len(df)} removed)")
        return df

    def _remove_emojis(self, text: str) -> str:
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(self.emoji_regex, "", text)
        return text

    def _replace_urls(self, text: str) -> str:
        return re.sub(r'https?://\S+|www\.\S+', ' URL ', text)

    def _replace_mentions(self, text: str) -> str:
        return re.sub(r'@\w+', ' PEOPLE ', text)

    def _extract_hashtags(self, text: str) -> Tuple[List[str], str]:
        hashtags = re.findall(r'#\w+', text)
        text_without = re.sub(r'#\w+', '', text)
        return hashtags, text_without

    def _standardize(self, text: str) -> str:
        return text.lower().replace('\n', ' ').replace('\r', ' ')

    def _clean_text(self, text: str) -> Tuple[str, List[str]]:
        text = self._standardize(text)
        text = self._remove_emojis(text)
        text = self._replace_urls(text)
        text = self._replace_mentions(text)
        hashtags, text = self._extract_hashtags(text)
        text = re.sub(r"[^\w\s!?\-']", "", text)
        text = self._remove_emojis(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, hashtags

    def _tokenize(self, cleaned_text: str) -> List[str]:
        doc = self.nlp(cleaned_text)
        return [token.text.lower() for token in doc if token.is_alpha]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        doc = self.nlp(" ".join(tokens))
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.lemma_ != ""
        ]

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords_set]

    def _preprocess(self, cleaned_text: str) -> List[str]:
        return self._remove_stopwords(self._lemmatize(self._tokenize(cleaned_text)))

    def load_and_clean(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = self._load_raw()
        df     = df_raw.copy()

        cleaned_results    = df[self.cfg.TEXT_COL].apply(self._clean_text)
        df["cleaned_text"] = cleaned_results.apply(lambda x: x[0])
        df["hashtags"]     = cleaned_results.apply(lambda x: x[1])

        url_count   = df[self.cfg.TEXT_COL].apply(
            lambda x: len(re.findall(r'https?://\S+|www\.\S+', x))
        ).sum()
        posts_w_url = df[self.cfg.TEXT_COL].str.contains(r'https?://|www\.', regex=True).sum()
        mention_count   = df[self.cfg.TEXT_COL].apply(
            lambda x: len(re.findall(r'@\w+', x))
        ).sum()
        posts_w_mention = df[self.cfg.TEXT_COL].str.contains(r'@\w+', regex=True).sum()
        hashtag_count   = df["hashtags"].apply(len).sum()
        posts_w_hashtag = (df["hashtags"].apply(len) > 0).sum()

        print(f"\n── Noise Element Counts ──")
        print(f"   URLs      : {url_count}  (in {posts_w_url} posts)")
        print(f"   @mentions : {mention_count}  (in {posts_w_mention} posts)")
        print(f"   #hashtags : {hashtag_count}  (in {posts_w_hashtag} posts)")

        df["tokens"]    = df["cleaned_text"].apply(self._preprocess)
        df["char_count"]        = df["cleaned_text"].apply(len)
        df["text_length"]       = df["cleaned_text"].apply(lambda x: len(x.split()))
        df["punct_count"]       = df["cleaned_text"].apply(
            lambda x: x.count('?') + x.count('!') + x.count('...')
        )
        df["question_count"]    = df["cleaned_text"].apply(lambda x: x.count('?'))
        df["exclamation_count"] = df["cleaned_text"].apply(lambda x: x.count('!'))
        df["ellipsis_count"]    = df["cleaned_text"].apply(lambda x: x.count('...'))
        df["text_nostop"]       = df["tokens"].apply(lambda t: " ".join(t))
        df["emoji_count"]       = df[self.cfg.TEXT_COL].apply(lambda x: len(emoji.emoji_list(x)))
        df["emoticon_count"]    = df[self.cfg.TEXT_COL].apply(
            lambda x: sum(len(re.findall(p, x, re.IGNORECASE)) for p in self.emoticon_patterns)
        )

        df["emoji_count_after"] = df["cleaned_text"].apply(lambda x: len(emoji.emoji_list(x)))
        before  = df["emoji_count"].sum()
        after   = df["emoji_count_after"].sum()
        removed = before - after
        print(f"\n── Emoji Removal Summary ──")
        print(f"   Before: {before}  |  After: {after}  |  Removed: {removed}")
        print("   ✓ All emojis removed" if after == 0 else f"   ⚠ {after} emojis remain")
        df.drop(columns=["emoji_count_after"], inplace=True)

        missing = df[self.cfg.LABEL_COL].isna().sum()
        if missing > 0:
            df = df.dropna(subset=[self.cfg.LABEL_COL]).reset_index(drop=True)

        print(f"\n[TextCleaner] Pipeline complete — {len(df)} rows ready for EDA")
        return df_raw, df

    def tokenize_public(self, text: str) -> List[str]:
        return self._tokenize(text)

    def lemmatize_public(self, tokens: List[str]) -> List[str]:
        return self._lemmatize(tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 3. EDA ANALYSIS  — one file per graph
# ══════════════════════════════════════════════════════════════════════════════
class EDAAnalysis:
    def __init__(self, cfg: Config, helper: PlotHelper, df: pd.DataFrame) -> None:
        self.cfg    = cfg
        self.helper = helper
        self.df     = df

    # ── helpers ───────────────────────────────────────────────────────────────
    def _make_autopct(self, values):
        def autopct(pct):
            count = int(round(pct * sum(values) / 100.0))
            return f"{pct:.1f}%\n(n={count})"
        return autopct

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 1 — Label Distribution  →  2 separate files
    # ──────────────────────────────────────────────────────────────────────────
    def graph_01(self) -> None:
        counts = self.df[self.cfg.LABEL_COL].value_counts()
        colors = label_palette(counts.index)

        # 01a — bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.barplot(x=counts.values, y=counts.index.astype(str),
                    palette=colors, ax=ax)
        ax.set_xlabel("Count")
        # ← no title
        for bar, val in zip(ax.patches, counts.values):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9)
        self.helper.save("01a_label_distribution_bar.png")

        # 01b — pie chart
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.pie(
            counts.values, labels=counts.index,
            autopct=self._make_autopct(counts.values),
            colors=colors,
            startangle=140,
        )
        # ← no title
        self.helper.save("01b_label_distribution_pie.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 2 — Text Length Analysis  →  3 separate files
    # ──────────────────────────────────────────────────────────────────────────
    def graph_02(self) -> None:
        df     = self.df
        labels = df[self.cfg.LABEL_COL].unique()
        colors = label_palette(labels)

        # 02a — histogram of word count
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
        fig.patch.set_facecolor("white")
        # ← no suptitle
        if len(labels) == 1:
            axes = [axes]
        for ax, label, color in zip(axes, labels, colors):
            ax.set_facecolor("white")
            subset = df[df[self.cfg.LABEL_COL] == label]["text_length"]
            counts_h, bins, patches = ax.hist(subset, bins=30, color=color,
                                            edgecolor="white", alpha=0.85)
            median_val = subset.median()
            ax.axvline(median_val, color="red", linestyle="--", linewidth=1.8,
                       label=f"Median = {median_val:.0f}")
            ax.legend(fontsize=9)
            for count, patch in zip(counts_h, patches):
                if count > 0:
                    ax.text(patch.get_x() + patch.get_width() / 2,
                            patch.get_height() + max(counts_h) * 0.01,
                            f"{int(count)}", ha="center", va="bottom", fontsize=8)
            ax.set_title(f"{label}", fontsize=11, fontweight="bold")   # sub-panel label kept
            ax.set_xlabel("Text Length (words)")
            ax.set_ylabel("Frequency")
        self.helper.save("02a_textlength_histogram_by_label.png")

        # shared boxplot helper
        def _boxplot(col, ylabel, fname):
            fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
            fig.patch.set_facecolor("white")
            # ← no suptitle
            if len(labels) == 1:
                axes = [axes]
            for ax, label, color in zip(axes, labels, colors):
                ax.set_facecolor("white")
                subset = df[df[self.cfg.LABEL_COL] == label][col]
                ax.boxplot(subset, patch_artist=True,
                           boxprops=dict(facecolor=color, color="gray"),
                           medianprops=dict(color="black", linewidth=2),
                           flierprops=dict(marker='o', markersize=3,
                                           markerfacecolor=color, alpha=0.4))
                mn     = subset.min();   q1 = subset.quantile(0.25)
                median = subset.median(); mean = subset.mean()
                q3     = subset.quantile(0.75); mx = subset.max()
                for val, lbl, offset in [
                    (mn,     f"Min: {mn:.0f}",       -0.32),
                    (q1,     f"Q1: {q1:.0f}",          0.32),
                    (median, f"Median: {median:.0f}",   0.32),
                    (mean,   f"Mean: {mean:.0f}",      -0.32),
                    (q3,     f"Q3: {q3:.0f}",           0.32),
                    (mx,     f"Max: {mx:.0f}",         -0.32),
                ]:
                    ax.text(1 + offset, val, lbl, ha="center", va="center", fontsize=8)
                ax.set_title(f"{ylabel} — {label}")   # sub-panel label kept
                ax.set_ylabel(ylabel)
                ax.set_xticks([])
            self.helper.save(fname)

        # 02b — word count boxplot
        _boxplot("text_length", "Text Length (words)", "02b_textlength_boxplot_by_label.png")

        # 02c — char count boxplot
        _boxplot("char_count", "Char Count", "02c_charcount_boxplot_by_label.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 3 — Punctuation Analysis  →  2 separate files
    # ──────────────────────────────────────────────────────────────────────────
    def graph_03(self) -> None:
        df         = self.df
        punct_cols = ['question_count', 'exclamation_count', 'ellipsis_count']
        sent_counts = df['cleaned_text'].apply(lambda t: max(len(sent_tokenize(t)), 1))

        def _build_norm_df():
            summary = []
            for label in df[self.cfg.LABEL_COL].unique():
                mask      = df[self.cfg.LABEL_COL] == label
                avg_sents = sent_counts[mask].mean()
                row = {"Label": label}
                for col in punct_cols:
                    avg = df[mask][col].mean()
                    row[col] = avg / avg_sents if avg_sents > 0 else 0
                summary.append(row)
            norm = pd.DataFrame(summary).set_index("Label")
            norm.columns = ["Question", "Exclamation", "Ellipsis"]
            return norm

        norm_df = _build_norm_df()

        # 03a — bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        # ← no title
        x      = np.arange(len(norm_df.columns))
        labels = norm_df.index.tolist()
        n      = len(labels)
        width  = 0.35
        colors = label_palette(labels)
        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(x + offset, norm_df.loc[label],
                            width=width, label=label, color=color, edgecolor="white")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(norm_df.columns, fontsize=11)
        ax.set_ylabel("Avg count per sentence")
        ax.set_xlabel("Punctuation type")
        ax.legend(title="Label")
        ax.set_ylim(0, norm_df.values.max() * 1.25)
        self.helper.save("03a_punctuation_normalized_bar.png")

        # 03b — table
        norm_df_rounded = _build_norm_df().round(4)
        fig, ax = plt.subplots(figsize=(8, 2 + len(norm_df_rounded) * 0.6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")
        # ← no title
        colors_rows = label_palette(norm_df_rounded.index)
        row_colors  = [[c] + ["#f9f9f9"] * len(norm_df_rounded.columns) for c in colors_rows]
        table = ax.table(
            cellText    = norm_df_rounded.reset_index().values,
            colLabels   = ["Label"] + list(norm_df_rounded.columns),
            cellLoc     = "center",
            loc         = "center",
            cellColours = row_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2)
        for j in range(len(norm_df_rounded.columns) + 1):
            table[0, j].set_text_props(fontweight="bold", color="white")
            table[0, j].set_facecolor("#4C72B0")
        self.helper.save("03b_punctuation_normalized_table.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 4 — Word Clouds per Label  →  1 file (one cloud per label, tiled)
    # ──────────────────────────────────────────────────────────────────────────
    def graph_04(self) -> None:
        df     = self.df
        labels = df[self.cfg.LABEL_COL].unique()
        n      = len(labels)
        cols   = min(3, n)
        rows   = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        fig.patch.set_facecolor("white")
        axes = np.array(axes).flatten()
        # ← no suptitle
        # Word-cloud colour: use single colour per label matching the label palette
        wc_colors = {
            "Healthy":   "YlOrBr",   # warm yellow/gold tones → close to #D8BE3D
            "Unhealthy": "Purples",  # purple tones → close to #463EBD
        }
        default_cmaps = ["Blues", "Reds", "Greens", "Oranges"]
        for i, label in enumerate(labels):
            tokens_filtered = (
                df[df[self.cfg.LABEL_COL] == label]["tokens"]
                .apply(lambda t: [w for w in t if w not in WORDCLOUD_STOPWORDS])
            )
            text = " ".join([" ".join(t) for t in tokens_filtered])
            if not text.strip():
                axes[i].axis("off")
                continue
            cmap = wc_colors.get(str(label), default_cmaps[i % len(default_cmaps)])
            wc = WordCloud(width=600, height=350, background_color="white",
                           colormap=cmap, max_words=100,
                           collocations=False).generate(text)
            axes[i].set_facecolor("white")
            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].axis("off")
            axes[i].set_title(str(label), fontsize=12, fontweight="bold")  # sub-panel label kept
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        self.helper.save("04_wordclouds_per_label.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 5 — Co-occurrence  →  1 file per label
    # ──────────────────────────────────────────────────────────────────────────
    def graph_05(self) -> None:
        df     = self.df
        labels = df[self.cfg.LABEL_COL].unique()

        def _cooccurrence(texts, top_n):
            co = Counter()
            for sentence in texts:
                words = list(set(sentence.split()))
                for pair in combinations(sorted(words), 2):
                    co[pair] += 1
            return co.most_common(top_n)

        import matplotlib.colors as mcolors

        def _cooc_gradient(label: str, n_colors: int):
            base_hex = LABEL_COLORS.get(str(label), "#888888")
            base_rgb = mcolors.to_rgb(base_hex)
            shades = []
            for i in range(n_colors):
                factor = 0.40 + 0.60 * (i / max(n_colors - 1, 1))
                shades.append(tuple(min(1.0, c * factor + (1 - factor)) for c in base_rgb))
            return list(reversed(shades))

        for label in labels:
            texts  = df[df[self.cfg.LABEL_COL] == label]["text_nostop"]
            pairs  = _cooccurrence(texts, self.cfg.TOP_N_COOC)
            if not pairs:
                continue
            pair_labels = [f"{a} & {b}" for (a, b), _ in pairs]
            counts      = [c for _, c in pairs]
            palette     = _cooc_gradient(label, len(counts))
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            sns.barplot(x=counts, y=pair_labels, palette=palette, ax=ax)
            # ← no title
            ax.set_xlabel("Co-occurrence count")
            for bar, val in zip(ax.patches, counts):
                ax.text(bar.get_width() + max(counts) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=9)
            ax.set_xlim(0, max(counts) * 1.12)
            fname = f"05_cooccurrence_{self.helper.safe_name(label)}.png"
            self.helper.save(fname)

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 6 — N-grams  →  3 separate files (unigrams / bigrams / trigrams)
    # ──────────────────────────────────────────────────────────────────────────
    def graph_06(self) -> None:
        df = self.df

        def _get_ngram_freq(texts, n):
            vectorizer = CountVectorizer(ngram_range=(n, n), min_df=2, max_df=0.95)
            X     = vectorizer.fit_transform(texts)
            freqs = X.toarray().sum(axis=0)
            result = pd.DataFrame({"ngram": vectorizer.get_feature_names_out(), "count": freqs})
            return result.sort_values("count", ascending=False).head(self.cfg.TOP_N_WORDS)

        def _label_gradient_palette(label: str, n_colors: int):
            """
            Return a list of n_colors shades derived from the label's base colour.
            Darkest shade for the highest bar, lightest for the lowest.
            """
            import matplotlib.colors as mcolors
            base_hex = LABEL_COLORS.get(str(label), "#888888")
            base_rgb = mcolors.to_rgb(base_hex)
            # Generate shades from 40% brightness (dark) to 100% (full colour)
            shades = []
            for i in range(n_colors):
                factor = 0.40 + 0.60 * (i / max(n_colors - 1, 1))
                shades.append(tuple(min(1.0, c * factor + (1 - factor)) for c in base_rgb))
            # Reverse so darkest = highest bar (first in sorted-descending list)
            return list(reversed(shades))

        def _plot_ngrams(n, title_prefix, filename):
            labels = df[self.cfg.LABEL_COL].unique()
            cols   = min(2, len(labels))
            rows   = (len(labels) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
            fig.patch.set_facecolor("white")
            axes = np.array(axes).flatten()
            # ← no suptitle
            for i, label in enumerate(labels):
                texts      = df[df[self.cfg.LABEL_COL] == label]["text_nostop"].dropna().astype(str)
                top_ngrams = _get_ngram_freq(texts, n)
                axes[i].set_facecolor("white")
                if top_ngrams.empty:
                    axes[i].axis("off")
                    continue
                palette = _label_gradient_palette(label, len(top_ngrams))
                sns.barplot(data=top_ngrams, x="count", y="ngram",
                            palette=palette, ax=axes[i])
                axes[i].set_title(str(label), fontsize=11, fontweight="bold")
                axes[i].set_xlabel("Frequency")
                axes[i].set_ylabel("")
                for bar, val in zip(axes[i].patches, top_ngrams["count"]):
                    axes[i].text(bar.get_width() + max(top_ngrams["count"]) * 0.01,
                                 bar.get_y() + bar.get_height() / 2,
                                 str(val), va="center", fontsize=8)
                axes[i].set_xlim(0, max(top_ngrams["count"]) * 1.12)
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
            self.helper.save(filename)

        _plot_ngrams(1, "Common Words (Unigrams)", "06a_common_words_per_label.png")
        _plot_ngrams(2, "Bigrams",                 "06b_bigrams_per_label.png")
        _plot_ngrams(3, "Trigrams",                "06c_trigrams_per_label.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 7 — Category Distribution  →  2 separate files
    # ──────────────────────────────────────────────────────────────────────────
    def graph_07(self) -> None:
        counts = self.df["category"].value_counts()

        # 07a — bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.barplot(x=counts.values, y=counts.index.astype(str), palette="Set2", ax=ax)
        ax.set_xlabel("Count")
        # ← no title
        for bar, val in zip(ax.patches, counts.values):
            ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9)
        self.helper.save("07a_category_distribution_bar.png")

        # 07b — pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.pie(counts.values, labels=counts.index,
               autopct=self._make_autopct(counts.values),
               colors=sns.color_palette("Set2", len(counts)),
               startangle=140)
        # ← no title
        self.helper.save("07b_category_distribution_pie.png")

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 8 — Category × Label Heatmap  →  1 file
    # ──────────────────────────────────────────────────────────────────────────
    def graph_08(self) -> None:
        cross = pd.crosstab(self.df["category"], self.df[self.cfg.LABEL_COL])
        cross.loc["Total"] = cross.sum()
        cross["Total"]     = cross.sum(axis=1)
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, annot_kws={"size": 11, "weight": "bold"})
        # ← no title
        ax.set_xlabel("Mental State")
        ax.set_ylabel("Category")
        self.helper.save("08_category_label_heatmap.png")

    def graph_08b(self) -> None:
        df = self.df
        for category in df["category"].unique():
            subset = df[df["category"] == category]
            healthy_words   = " ".join(subset[subset["mental_state"] == "Healthy"]["text_nostop"])
            unhealthy_words = " ".join(subset[subset["mental_state"] == "Unhealthy"]["text_nostop"])
            h_freq = Counter(healthy_words.split()).most_common(15)
            u_freq = Counter(unhealthy_words.split()).most_common(15)
            print(f"\n=== {category} ===")
            print("Healthy top words:",   h_freq)
            print("Unhealthy top words:", u_freq)

    # ──────────────────────────────────────────────────────────────────────────
    # GRAPH 10 — Emoji & Emoticon  →  5 separate files
    # ──────────────────────────────────────────────────────────────────────────
    def graph_10(self) -> None:
        df = self.df

        # 10a — emoji count histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.hist(df["emoji_count"], bins=20, color="#F4A460", edgecolor="white")
        median = df["emoji_count"].median()
        ax.axvline(median, color="navy", linestyle=":", label=f"Median={median:.4f}")
        # ← no title
        ax.set_xlabel("Emojis per text")
        ax.legend()
        for patch in ax.patches:
            h = patch.get_height()
            if h > 0:
                ax.text(patch.get_x() + patch.get_width() / 2,
                        h + 0.5, str(int(h)), ha="center", va="bottom", fontsize=8)
        self.helper.save("10a_emoji_count_histogram.png")

        # 10b — emoticon count histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.hist(df["emoticon_count"], bins=20, color="#87CEEB", edgecolor="white")
        median = df["emoticon_count"].median()
        ax.axvline(median, color="navy", linestyle=":", label=f"Median={median:.4f}")
        # ← no title
        ax.set_xlabel("Emoticons per text")
        ax.legend()
        for patch in ax.patches:
            h = patch.get_height()
            if h > 0:
                ax.text(patch.get_x() + patch.get_width() / 2,
                        h + 0.5, str(int(h)), ha="center", va="bottom", fontsize=8)
        self.helper.save("10b_emoticon_count_histogram.png")

        # 10c — emoji presence pie
        emoji_presence = (df["emoji_count"] > 0).value_counts()
        no_emoji  = emoji_presence.get(False, 0)
        has_emoji = emoji_presence.get(True,  0)
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.pie([no_emoji, has_emoji],
               labels=[f"No Emoji\n(n={no_emoji})", f"Contains Emoji\n(n={has_emoji})"],
               autopct="%1.2f%%", colors=["#d3d3d3", "#F4A460"], startangle=90)
        # ← no title
        self.helper.save("10c_emoji_presence_pie.png")

        # 10d — avg emoji count by label
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        avg = df.groupby(self.cfg.LABEL_COL)["emoji_count"].mean().sort_values(ascending=False)
        colors = label_palette(avg.index)
        sns.barplot(x=avg.values, y=avg.index.astype(str), palette=colors, ax=ax)
        # ← no title
        ax.set_xlabel("Avg count")
        for bar, val in zip(ax.patches, avg.values):
            ax.text(bar.get_width() + avg.values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, avg.values.max() * 1.15)
        self.helper.save("10d_avg_emoji_by_label.png")

        # 10e — avg emoticon count by label
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        avg = df.groupby(self.cfg.LABEL_COL)["emoticon_count"].mean().sort_values(ascending=False)
        colors = label_palette(avg.index)
        sns.barplot(x=avg.values, y=avg.index.astype(str), palette=colors, ax=ax)
        # ← no title
        ax.set_xlabel("Avg count")
        for bar, val in zip(ax.patches, avg.values):
            ax.text(bar.get_width() + avg.values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlim(0, avg.values.max() * 1.15)
        self.helper.save("10e_avg_emoticon_by_label.png")

    # ──────────────────────────────────────────────────────────────────────────
    # run_all
    # ──────────────────────────────────────────────────────────────────────────
    def run_all(self) -> None:
        print("\n── Running all EDA graphs ──")
        self.graph_01()
        self.graph_02()
        self.graph_03()
        self.graph_04()
        self.graph_05()
        self.graph_06()
        self.graph_07()
        self.graph_08()
        self.graph_08b()
        self.graph_10()
        print("\n── All EDA graphs complete ──")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

cleaner    = TextCleaner(cfg)
df_raw, df = cleaner.load_and_clean()

print("\n── Emoji & Emoticon Diagnostics ──")
print(df["emoji_count"].value_counts())
print(df["emoticon_count"].value_counts())
print(f"Posts with any emoji:    {(df['emoji_count'] > 0).sum()}")
print(f"Posts with any emoticon: {(df['emoticon_count'] > 0).sum()}")

eda = EDAAnalysis(cfg, helper, df)
eda.run_all()

df.to_csv(f"{cfg.OUTPUT_DIR}/french_cleaned.csv", index=False, encoding="utf-8-sig")

all_emojis = []
for text in df_raw[cfg.TEXT_COL]:
    all_emojis.extend([item['emoji'] for item in emoji.emoji_list(str(text))])

emoji_counts   = Counter(all_emojis)
emoji_txt_path = os.path.join(cfg.OUTPUT_DIR, "emojis_hashtags_found.txt")

with open(emoji_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Emojis found in ORIGINAL dataset ({len(emoji_counts)} unique)\n")
    f.write("=" * 45 + "\n")
    for em, count in emoji_counts.most_common():
        f.write(f"  {em}  →  {count} times\n")

all_hashtags = Counter([tag for tags in df["hashtags"] for tag in tags])

with open(emoji_txt_path, "a", encoding="utf-8") as f:
    f.write(f"\n\nHashtags found in dataset ({len(all_hashtags)} unique)\n")
    f.write("=" * 45 + "\n")
    for tag, count in all_hashtags.most_common():
        f.write(f"  {tag}  →  {count} times\n")

print(f"\n── Hashtags found ({len(all_hashtags)} unique) ──")
for tag, count in all_hashtags.most_common():
    print(f"   {tag}  →  {count} times")

print(f"\n── Emojis found in ORIGINAL dataset ({len(emoji_counts)} unique) ──")
for em, count in emoji_counts.most_common():
    print(f"   {em}  →  {count} times")

print("\n── Lemmatisation Examples ──\n")
for i in range(3):
    raw        = df_raw[cfg.TEXT_COL].iloc[i]
    cleaned    = df["cleaned_text"].iloc[i]
    tokens_raw = cleaner.tokenize_public(cleaned)
    lemmas     = cleaner.lemmatize_public(tokens_raw)
    final      = df["tokens"].iloc[i]
    print(f"POST {i+1}")
    print(f"  Original : {raw[:120]}")
    print(f"  Cleaned  : {cleaned[:120]}")
    print(f"  Tokens   : {tokens_raw[:10]}")
    print(f"  Lemmas   : {lemmas[:10]}")
    print(f"  Final    : {final[:10]}")
    print()

print("=" * 55)
print("  FRENCH EDA PIPELINE — COMPLETE")
print("=" * 55)

saved = [f for f in os.listdir(cfg.OUTPUT_DIR) if f.endswith(".png")]
print(f"\n📊 {len(saved)} plots saved to '{cfg.OUTPUT_DIR}/':")
for f in sorted(saved):
    print(f"   • {f}")

print(f"\n📄 Cleaned CSV : {cfg.OUTPUT_DIR}/french_cleaned.csv")
print(f"📄 Emoji log   : {emoji_txt_path}")
print("\n✅ All done!")