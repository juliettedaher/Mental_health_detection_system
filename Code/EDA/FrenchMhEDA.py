"""
mental_health_EDA.py
====================
Exploratory Data Analysis pipeline for a French mental-health social-media dataset.

Pipeline overview
-----------------
0. Config         – central settings (paths, column names, visual style)
1. PlotHelper     – save figures & sanitise filenames
2. DataLoader     – load CSV, filter French rows
3. TextCleaner    – clean / tokenise / lemmatise / extract features
4. Analysis classes – one per requirement / chart group
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import string
import warnings
from collections import Counter
from itertools import combinations

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import spacy
import emoji

# ── Typing ────────────────────────────────────────────────────────────────────
from typing import List, Tuple, Set, Dict, Optional

# ── NLP ───────────────────────────────────────────────────────────────────────
from nltk.tokenize import sent_tokenize

# ── NLTK resource downloads (silent; failures are non-fatal) ──────────────────
import nltk
try:
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet',   quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

warnings.filterwarnings("ignore")
print("All imports OK")


# ═══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    """
    Central place for every tuneable constant in the pipeline.
    Changing a value here propagates automatically to every class that
    receives a cfg argument — no hunting for magic strings.
    """
    CSV_PATH       = r"C:\Users\Admin\Documents\FYP\french dataset\Dataset\french_data.csv"
    OUTPUT_DIR     = r"MyResults"

    LANGUAGE_COL   = "language"
    LANGUAGE_VALUE = "French"
    TEXT_COL       = "text"
    LABEL_COL      = "mental_state"

    # ── Visual style ──────────────────────────────────────────────────────────
    BG      = "#F9F9F9"
    DPI     = 150
    PALETTE = "Set2"

    # ── Analysis limits ───────────────────────────────────────────────────────
    TOP_N_WORDS = 20
    TOP_N_COOC  = 20


cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(f"Config ready — output folder: '{cfg.OUTPUT_DIR}'")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PLOT HELPER
# ═══════════════════════════════════════════════════════════════════════════════
class PlotHelper:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        plt.rcParams.update({
            "figure.facecolor" : cfg.BG,
            "axes.facecolor"   : cfg.BG,
            "axes.spines.top"  : False,
            "axes.spines.right": False,
            "font.size"        : 11,
        })

    def save(self, filename: str) -> str:
        """Apply tight layout, save the figure to OUTPUT_DIR, close it, return the path."""
        path = os.path.join(self.cfg.OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=self.cfg.DPI, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {filename}")
        return path

    @staticmethod
    def safe_name(text: str) -> str:
        """Replace OS-illegal characters in a string so it can be used as a filename."""
        return re.sub(r'[\\/*?"<>|]+', "_", str(text)).strip()


helper = PlotHelper(cfg)
print("PlotHelper ready")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════
class DataLoader:
    """
    Loads the raw CSV and returns only the French-language rows.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        """Read the CSV (UTF-8 with BOM), filter to French rows, reset index."""
        df = pd.read_csv(self.cfg.CSV_PATH, encoding="utf-8-sig")
        print(f"[DataLoader] Total rows loaded : {len(df)}")

        mask = (
            df[self.cfg.LANGUAGE_COL].str.strip().str.lower()
            == self.cfg.LANGUAGE_VALUE.lower()
        )
        df = df[mask].copy().reset_index(drop=True)
        print(f"[DataLoader] French rows kept  : {len(df)}")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TEXT CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

PRONOUNS = {
    "je", "j", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "me", "moi", "te", "toi", "se",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs"
}

EXTRA_REMOVE = {
    "le", "la", "les",
    "un", "une", "des",
    "du", "au", "aux",
    "de", "à", "en", "dans", "sur", "avec", "pour", "par", "sans", "chez",
    "et", "ou", "mais", "donc", "or", "ni", "car",
    "que", "qui", "quand", "lorsque", "comme", "puisque", "quoique", "quoi",
    "si", "afin", "bien", "pendant", "avant", "après", "depuis", "jusqu", "malgré",
    "chaque", "tous", "toutes", "tout", "toute", "tous", "toutes",
    "ce", "cet", "cette", "ces",
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
    "résilience", "guérison"
}

STOPWORDS = (PRONOUNS | EXTRA_REMOVE | NOISE | REMOVE_VERBS) - KEEP_WORDS


class TextCleaner:
    """
    All text cleaning, tokenisation, lemmatisation, and feature extraction
    needed to convert raw post text into analysis-ready columns.
    """

    def __init__(self) -> None:
        self.nlp           = nlp
        self.stopwords_set: Set[str] = STOPWORDS

        self.emoji_regex = (
            r'[\U0001F600-\U0001F64F]|'
            r'[\U0001F300-\U0001F5FF]|'
            r'[\U0001F680-\U0001F6FF]|'
            r'[\U00002600-\U000026FF]|'
            r'[\U00002700-\U000027BF]|'
            r'[\U0001F900-\U0001F9FF]|'
            r'[\U0001FA00-\U0001FA6F]|'
            r'[\U0001FA70-\U0001FAFF]'
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

    def remove_emojis(self, text: str) -> str:
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(self.emoji_regex, "", text)
        return text

    def replace_urls(self, text: str) -> str:
        return re.sub(r'https?://\S+|www\.\S+', ' URL ', text)

    def replace_mentions(self, text: str) -> str:
        return re.sub(r'@\w+', ' PEOPLE ', text)

    def extract_hashtags(self, text: str) -> Tuple[List[str], str]:
        hashtags = re.findall(r'#\w+', text)
        text_without_hashtags = re.sub(r'#\w+', '', text)
        return hashtags, text_without_hashtags

    def standardize_text(self, text: str) -> str:
        return text.lower().replace('\n', ' ').replace('\r', ' ')

    def tokenize(self, cleaned_text: str) -> List[str]:
        doc = self.nlp(cleaned_text)
        return [token.text.lower() for token in doc if token.is_alpha]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        doc = self.nlp(" ".join(tokens))
        return [token.lemma_.lower() for token in doc if token.is_alpha]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords_set]

    def clean_text(self, text: str) -> Tuple[str, List[str]]:
        text = self.standardize_text(text)
        text = self.remove_emojis(text)
        text = self.replace_urls(text)
        text = self.replace_mentions(text)
        hashtags, text = self.extract_hashtags(text)
        text = re.sub(r"[^\w\s!?\-']", "", text)
        text = self.remove_emojis(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, hashtags

    def preprocess(self, cleaned_text: str) -> List[str]:
        tokens = self.tokenize(cleaned_text)
        lemmas = self.lemmatize(tokens)
        return self.remove_stopwords(lemmas)

    def fit_transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        df = df.copy()

        cleaned_results    = df[text_col].apply(self.clean_text)
        df["cleaned_text"] = cleaned_results.apply(lambda x: x[0])
        df["hashtags"]     = cleaned_results.apply(lambda x: x[1])

        df["tokens"] = df["cleaned_text"].apply(self.preprocess)

        df["char_count"]        = df["cleaned_text"].apply(len)
        df["text_length"]       = df["cleaned_text"].apply(lambda x: len(x.split()))
        df["punct_count"]       = df["cleaned_text"].apply(
            lambda x: x.count('?') + x.count('!') + x.count('...')
        )
        df["question_count"]    = df["cleaned_text"].apply(lambda x: x.count('?'))
        df["exclamation_count"] = df["cleaned_text"].apply(lambda x: x.count('!'))
        df["ellipsis_count"]    = df["cleaned_text"].apply(lambda x: x.count('...'))

        df["text_nostop"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

        df["emoji_count"]    = df[text_col].apply(lambda x: len(emoji.emoji_list(x)))
        df["emoticon_count"] = df[text_col].apply(
            lambda x: sum(
                len(re.findall(p, x, re.IGNORECASE))
                for p in self.emoticon_patterns
            )
        )

        print(f"[TextCleaner] Cleaned & tokenized {len(df)} rows")
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EXECUTION — load & clean data
# ═══════════════════════════════════════════════════════════════════════════════
df_raw  = DataLoader(cfg).load()
cleaner = TextCleaner()
df      = cleaner.fit_transform(df_raw, cfg.TEXT_COL)

print("\n── Emoji & Emoticon Diagnostics ──")
print(df["emoji_count"].value_counts())
print(df["emoticon_count"].value_counts())
print(f"Posts with any emoji:    {(df['emoji_count'] > 0).sum()}")
print(f"Posts with any emoticon: {(df['emoticon_count'] > 0).sum()}")

df.head(3)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Label Distribution
# ═══════════════════════════════════════════════════════════════════════════════
class LabelDistribution:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        counts = df[self.cfg.LABEL_COL].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Label Distribution — mental_state", fontsize=14, fontweight="bold")

        sns.barplot(x=counts.values, y=counts.index.astype(str),
                    palette=self.cfg.PALETTE, ax=axes[0])
        axes[0].set_xlabel("Count")
        axes[0].set_title("Count per Label")
        for bar, val in zip(axes[0].patches, counts.values):
            axes[0].text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9,
            )

        def make_autopct(values):
            def autopct(pct):
                total = sum(values)
                count = int(round(pct * total / 100.0))
                return f"{pct:.1f}%\n(n={count})"
            return autopct

        axes[1].pie(
            counts.values,
            labels=counts.index,
            autopct=make_autopct(counts.values),
            colors=sns.color_palette(self.cfg.PALETTE, len(counts)),
            startangle=140,
        )
        axes[1].set_title("Proportion per Label")

        return self.helper.save("01_label_distribution.png")


LabelDistribution(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Text Length Analysis
# ═══════════════════════════════════════════════════════════════════════════════
class TextLengthAnalysis:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> None:
        labels = df[self.cfg.LABEL_COL].unique()
        colors = sns.color_palette(self.cfg.PALETTE, len(labels))

        # ── Plot 2a: Histogram with median line ───────────────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
        fig.suptitle("Text Length Distribution by Label", fontsize=14, fontweight="bold")

        if len(labels) == 1:
            axes = [axes]

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["text_length"]

            counts, bins, patches = ax.hist(
                subset,
                bins=30,
                color=color,
                edgecolor="white",
                alpha=0.85
            )

            # ── Median line ───────────────────────────────────────────────────
            median_val = subset.median()
            ax.axvline(
                median_val,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label=f"Median = {median_val:.0f}"
            )
            ax.legend(fontsize=9)

            # ── Bar value annotations ─────────────────────────────────────────
            for count, patch in zip(counts, patches):
                if count > 0:
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        patch.get_height() + max(counts) * 0.01,
                        f"{int(count)}",
                        ha="center",
                        va="bottom",
                        fontsize=8
                    )

            ax.set_title(f"{label}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Text Length")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        self.helper.save("02a_textlength_histogram_by_label.png")

        # ── Plot 2b: Boxplot ───────────────────────────────────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
        fig.suptitle("Text Length Boxplot by Label", fontsize=14, fontweight="bold")

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["text_length"]

            ax.boxplot(
                subset,
                patch_artist=True,
                boxprops=dict(facecolor=color, color="gray"),
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor=color, alpha=0.4),
            )

            mn     = subset.min()
            q1     = subset.quantile(0.25)
            median = subset.median()
            mean   = subset.mean()
            q3     = subset.quantile(0.75)
            mx     = subset.max()

            for val, lbl, offset in [
                (mn,     f"Min: {mn:.0f}",       -0.32),
                (q1,     f"Q1: {q1:.0f}",         0.32),
                (median, f"Median: {median:.0f}",  0.32),
                (mean,   f"Mean: {mean:.0f}",     -0.32),
                (q3,     f"Q3: {q3:.0f}",          0.32),
                (mx,     f"Max: {mx:.0f}",        -0.32),
            ]:
                ax.text(1 + offset, val, lbl, ha="center", va="center", fontsize=8)

            ax.set_title(f"Text Length — {label}")
            ax.set_ylabel("Text Length")
            ax.set_xticks([])

        plt.tight_layout()
        self.helper.save("02b_textlength_boxplot_by_label.png")

        # ── Plot 2c: Char Count Boxplot ────────────────────────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
        fig.suptitle("Char Count Distribution by Label", fontsize=14, fontweight="bold")

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["char_count"]

            ax.boxplot(
                subset,
                patch_artist=True,
                boxprops=dict(facecolor=color, color="gray"),
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor=color, alpha=0.4),
            )

            mn     = subset.min()
            q1     = subset.quantile(0.25)
            median = subset.median()
            mean   = subset.mean()
            q3     = subset.quantile(0.75)
            mx     = subset.max()

            for val, lbl, offset in [
                (mn,     f"Min: {mn:.0f}",       -0.32),
                (q1,     f"Q1: {q1:.0f}",         0.32),
                (median, f"Median: {median:.0f}",  0.32),
                (mean,   f"Mean: {mean:.0f}",     -0.32),
                (q3,     f"Q3: {q3:.0f}",          0.32),
                (mx,     f"Max: {mx:.0f}",        -0.32),
            ]:
                ax.text(1 + offset, val, lbl, ha="center", va="center", fontsize=8)

            ax.set_title(f"Char Count — {label}")
            ax.set_ylabel("Char Count")
            ax.set_xticks([])

        plt.tight_layout()
        self.helper.save("02c_charcount_boxplot_by_label.png")


TextLengthAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 3a — Punctuation Analysis
# ═══════════════════════════════════════════════════════════════════════════════
class PunctuationAnalysis:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        punct_cols = ['question_count', 'exclamation_count', 'ellipsis_count']

        sentence_counts = df['cleaned_text'].apply(
            lambda t: max(len(sent_tokenize(t)), 1)
        )

        summary = []
        for label in df[self.cfg.LABEL_COL].unique():
            mask      = df[self.cfg.LABEL_COL] == label
            avg_sents = sentence_counts[mask].mean()
            row       = {"Label": label}
            for col in punct_cols:
                avg_count = df[mask][col].mean()
                row[col]  = avg_count / avg_sents if avg_sents > 0 else 0
            summary.append(row)

        norm_df         = pd.DataFrame(summary).set_index("Label")
        norm_df.columns = ["Question", "Exclamation", "Ellipsis"]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Punctuation Usage Normalized by Sentence Length",
                     fontsize=14, fontweight="bold")

        x      = np.arange(len(norm_df.columns))
        labels = norm_df.index.tolist()
        n      = len(labels)
        width  = 0.35
        colors = sns.color_palette(self.cfg.PALETTE, n)

        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(x + offset, norm_df.loc[label],
                            width=width, label=label,
                            color=color, edgecolor="white")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(norm_df.columns, fontsize=11)
        ax.set_ylabel("Avg count per sentence")
        ax.set_xlabel("Punctuation type")
        ax.legend(title="Label")
        ax.set_ylim(0, norm_df.values.max() * 1.25)

        return self.helper.save("03_punctuation_normalized.png")


PunctuationAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 3b — Punctuation Table
# ═══════════════════════════════════════════════════════════════════════════════
class PunctuationTable:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        punct_cols      = ['question_count', 'exclamation_count', 'ellipsis_count']
        sentence_counts = df['cleaned_text'].apply(
            lambda t: max(len(sent_tokenize(t)), 1)
        )

        summary = []
        for label in df[self.cfg.LABEL_COL].unique():
            mask      = df[self.cfg.LABEL_COL] == label
            avg_sents = sentence_counts[mask].mean()
            row       = {"Label": label}
            for col in punct_cols:
                avg_count = df[mask][col].mean()
                row[col]  = round(avg_count / avg_sents if avg_sents > 0 else 0, 4)
            summary.append(row)

        norm_df         = pd.DataFrame(summary).set_index("Label")
        norm_df.columns = ["Question", "Exclamation", "Ellipsis"]

        fig, ax = plt.subplots(figsize=(8, 2 + len(norm_df) * 0.6))
        ax.axis("off")
        fig.suptitle("Punctuation Usage Normalized by Sentence Length",
                     fontsize=13, fontweight="bold", y=1.02)

        colors_rows = sns.color_palette(self.cfg.PALETTE, len(norm_df))
        row_colors  = [[c] + ["#f9f9f9"] * len(norm_df.columns) for c in colors_rows]

        table = ax.table(
            cellText    = norm_df.reset_index().values,
            colLabels   = ["Label"] + list(norm_df.columns),
            cellLoc     = "center",
            loc         = "center",
            cellColours = row_colors,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2)

        for j in range(len(norm_df.columns) + 1):
            table[0, j].set_text_props(fontweight="bold", color="white")
            table[0, j].set_facecolor("#4C72B0")

        return self.helper.save("03b_punctuation_table.png")


PunctuationTable(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 4 — Word Clouds per Label
# ═══════════════════════════════════════════════════════════════════════════════
class WordCloudAnalysis:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> list:
        labels = df[self.cfg.LABEL_COL].unique()
        n      = len(labels)
        cols   = min(3, n)
        rows   = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = np.array(axes).flatten()
        fig.suptitle("Word Clouds per Label", fontsize=15, fontweight="bold")

        cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges", "YlOrBr"]

        for i, label in enumerate(labels):
            text = " ".join(df[df[self.cfg.LABEL_COL] == label]["text_nostop"])
            if not text.strip():
                axes[i].axis("off")
                continue

            wc = WordCloud(
                width=600, height=350,
                background_color="white",
                colormap=cmaps[i % len(cmaps)],
                max_words=100,
                collocations=False,
            ).generate(text)

            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].axis("off")
            axes[i].set_title(str(label), fontsize=12, fontweight="bold")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        path = self.helper.save("04_wordclouds_per_label.png")
        return [path]


WordCloudAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 5 — Co-occurrence Analysis
# ═══════════════════════════════════════════════════════════════════════════════
class CoOccurrenceAnalysis:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def _cooccurrence(self, texts, top_n):
        co = Counter()
        for sentence in texts:
            words = list(set(sentence.split()))
            for pair in combinations(sorted(words), 2):
                co[pair] += 1
        return co.most_common(top_n)

    def analyse(self, df: pd.DataFrame) -> list:
        paths  = []
        labels = df[self.cfg.LABEL_COL].unique()

        for label in labels:
            texts = df[df[self.cfg.LABEL_COL] == label]["text_nostop"]
            pairs = self._cooccurrence(texts, self.cfg.TOP_N_COOC)
            if not pairs:
                continue

            pair_labels = [f"{a} & {b}" for (a, b), _ in pairs]
            counts      = [c for _, c in pairs]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = sns.barplot(x=counts, y=pair_labels, palette="mako", ax=ax)
            ax.set_title(f"Top Word Co-occurrences — {label}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Co-occurrence count")

            for bar, val in zip(ax.patches, counts):
                ax.text(
                    bar.get_width() + max(counts) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9,
                )
            ax.set_xlim(0, max(counts) * 1.12)

            fname = f"05_cooccurrence_{self.helper.safe_name(label)}.png"
            paths.append(self.helper.save(fname))

        return paths


CoOccurrenceAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 6 — Common Words / Bigrams / Trigrams
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.feature_extraction.text import CountVectorizer


class CommonWordsAnalysis:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def _get_ngram_freq(self, texts, n):
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            min_df=2,
            max_df=0.95
        )
        X     = vectorizer.fit_transform(texts)
        freqs = X.toarray().sum(axis=0)
        df    = pd.DataFrame({
            "ngram": vectorizer.get_feature_names_out(),
            "count": freqs
        })
        return df.sort_values("count", ascending=False).head(self.cfg.TOP_N_WORDS)

    def _plot_ngrams(self, df: pd.DataFrame, n: int, title_prefix: str, filename: str) -> str:
        labels = df[self.cfg.LABEL_COL].unique()
        cols   = min(2, len(labels))
        rows   = (len(labels) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
        axes = np.array(axes).flatten()

        fig.suptitle(f"Top {self.cfg.TOP_N_WORDS} {title_prefix} per Label",
                     fontsize=14, fontweight="bold")

        for i, label in enumerate(labels):
            texts      = df[df[self.cfg.LABEL_COL] == label]["text_nostop"].dropna().astype(str)
            top_ngrams = self._get_ngram_freq(texts, n)

            if top_ngrams.empty:
                axes[i].axis("off")
                continue

            sns.barplot(
                data=top_ngrams,
                x="count",
                y="ngram",
                palette="rocket",
                ax=axes[i]
            )

            axes[i].set_title(str(label), fontsize=11, fontweight="bold")
            axes[i].set_xlabel("Frequency")
            axes[i].set_ylabel("")

            for bar, val in zip(axes[i].patches, top_ngrams["count"]):
                axes[i].text(
                    bar.get_width() + max(top_ngrams["count"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    va="center",
                    fontsize=8,
                )

            axes[i].set_xlim(0, max(top_ngrams["count"]) * 1.12)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        return self.helper.save(filename)

    def analyse(self, df: pd.DataFrame) -> list:
        return [
            self._plot_ngrams(df, 1, "Common Words", "06_common_words_per_label.png"),
            self._plot_ngrams(df, 2, "Bigrams",       "06_bigrams_per_label.png"),
            self._plot_ngrams(df, 3, "Trigrams",      "06_trigrams_per_label.png"),
        ]


CommonWordsAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 7 — Category Distribution
# ═══════════════════════════════════════════════════════════════════════════════
class CategoryDistribution:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        counts = df["category"].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Distribution by Mental Health Category", fontsize=14, fontweight="bold")

        sns.barplot(x=counts.values, y=counts.index.astype(str),
                    palette="Set2", ax=axes[0])
        axes[0].set_xlabel("Count")
        axes[0].set_title("Count per Category")
        for bar, val in zip(axes[0].patches, counts.values):
            axes[0].text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", fontsize=9)

        def make_autopct(values):
            def autopct(pct):
                total = sum(values)
                count = int(round(pct * total / 100.0))
                return f"{pct:.1f}%\n(n={count})"
            return autopct

        axes[1].pie(counts.values, labels=counts.index,
                    autopct=make_autopct(counts.values),
                    colors=sns.color_palette("Set2", len(counts)),
                    startangle=140)
        axes[1].set_title("Proportion per Category")

        return self.helper.save("07_category_distribution.png")


CategoryDistribution(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 8 — Category × Label Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
class CategoryLabelHeatmap:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        cross = pd.crosstab(df["category"], df[self.cfg.LABEL_COL])

        cross.loc["Total"] = cross.sum()
        cross["Total"]     = cross.sum(axis=1)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title("Category × Mental State Heatmap",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Mental State")
        ax.set_ylabel("Category")

        return self.helper.save("08_category_label_heatmap.png")


CategoryLabelHeatmap(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 8b — Per-Category Healthy vs Unhealthy Word Comparison
# ═══════════════════════════════════════════════════════════════════════════════
for category in df["category"].unique():
    subset = df[df["category"] == category]

    healthy_words   = " ".join(subset[subset["mental_state"] == "Healthy"]["text_nostop"])
    unhealthy_words = " ".join(subset[subset["mental_state"] == "Unhealthy"]["text_nostop"])

    healthy_freq   = Counter(healthy_words.split()).most_common(15)
    unhealthy_freq = Counter(unhealthy_words.split()).most_common(15)

    print(f"\n=== {category} ===")
    print("Healthy top words:",   healthy_freq)
    print("Unhealthy top words:", unhealthy_freq)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 10 — Emoji & Emoticon Analysis  (fixed: pie chart now renders correctly)
# ═══════════════════════════════════════════════════════════════════════════════
class EmojiEmoticonAnalysis:
    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        # 2 rows × 3 cols so the pie chart has its own cell in row 0
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Emoji & Emoticon Analysis\n"
            "⚠️ <1% of posts contain emojis — results not statistically meaningful",
            fontsize=13, fontweight="bold"
        )

        # ── Row 0, cols 0-1: distribution histograms ──────────────────────────
        for col, ax, color, title, xlabel in [
            ("emoji_count",    axes[0, 0], "#F4A460", "Emoji Count Distribution",    "Emojis per text"),
            ("emoticon_count", axes[0, 1], "#87CEEB", "Emoticon Count Distribution", "Emoticons per text"),
        ]:
            ax.hist(df[col], bins=20, color=color, edgecolor="white")
            mean   = df[col].mean()
            median = df[col].median()
            ax.axvline(mean,   color="red",  linestyle="--", label=f"Mean={mean:.4f}")
            ax.axvline(median, color="navy", linestyle=":",  label=f"Median={median:.4f}")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.legend()
            for patch in ax.patches:
                h = patch.get_height()
                if h > 0:
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        h + 0.5, str(int(h)),
                        ha="center", va="bottom", fontsize=8,
                    )

        # ── Row 0, col 2: emoji presence pie chart ────────────────────────────
        ax_pie         = axes[0, 2]
        emoji_presence = (df["emoji_count"] > 0).value_counts()
        no_emoji       = emoji_presence.get(False, 0)
        has_emoji      = emoji_presence.get(True,  0)
        sizes          = [no_emoji, has_emoji]
        pie_labels     = [f"No Emoji\n(n={no_emoji})", f"Contains Emoji\n(n={has_emoji})"]

        ax_pie.pie(
            sizes,
            labels=pie_labels,
            autopct="%1.2f%%",
            colors=["#d3d3d3", "#F4A460"],
            startangle=90,
        )
        ax_pie.set_title("Emoji Presence Distribution")

        # ── Row 1, cols 0-1: avg per label ────────────────────────────────────
        for col, ax, title in [
            ("emoji_count",    axes[1, 0], "Avg Emoji Count by Label"),
            ("emoticon_count", axes[1, 1], "Avg Emoticon Count by Label"),
        ]:
            avg = df.groupby(self.cfg.LABEL_COL)[col].mean().sort_values(ascending=False)
            sns.barplot(x=avg.values, y=avg.index.astype(str),
                        palette=self.cfg.PALETTE, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Avg count")
            for bar, val in zip(ax.patches, avg.values):
                ax.text(
                    bar.get_width() + avg.values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9,
                )
            ax.set_xlim(0, avg.values.max() * 1.15)

        # ── Row 1, col 2: unused cell ─────────────────────────────────────────
        axes[1, 2].axis("off")

        plt.tight_layout()
        return self.helper.save("10_emoji_emoticon.png")


EmojiEmoticonAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
df.to_csv(f"{cfg.OUTPUT_DIR}/french_cleaned.csv", index=False, encoding="utf-8-sig")

# ── Emojis from the ORIGINAL dataset → saved to text file ────────────────────
all_emojis = []
for text in df_raw[cfg.TEXT_COL]:
    all_emojis.extend([item['emoji'] for item in emoji.emoji_list(str(text))])

emoji_counts   = Counter(all_emojis)
emoji_txt_path = os.path.join(cfg.OUTPUT_DIR, "emojis_found.txt")

with open(emoji_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Emojis found in ORIGINAL dataset ({len(emoji_counts)} unique)\n")
    f.write("=" * 45 + "\n")
    for em, count in emoji_counts.most_common():
        f.write(f"  {em}  →  {count} times\n")

print(f"\n── Emojis found in ORIGINAL dataset ({len(emoji_counts)} unique) ──")
for em, count in emoji_counts.most_common():
    print(f"   {em}  →  {count} times")
print(f"\n📄 Emoji list saved to: {emoji_txt_path}")

print("=" * 55)
print("  FRENCH EDA PIPELINE — COMPLETE")
print("=" * 55)

saved = [f for f in os.listdir(cfg.OUTPUT_DIR) if f.endswith(".png")]
print(f"\n📊 {len(saved)} plots saved to '{cfg.OUTPUT_DIR}/':")
for f in sorted(saved):
    print(f"   • {f}")

print(f"\n📄 Cleaned CSV  : {cfg.OUTPUT_DIR}/french_cleaned.csv")
print(f"📄 Emoji list   : {emoji_txt_path}")
print("\n✅ All done!")