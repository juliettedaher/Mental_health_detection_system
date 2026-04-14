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
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# ── NLTK resource downloads (silent; failures are non-fatal) ──────────────────
import nltk
try:
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
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
    receives a `cfg` argument — no hunting for magic strings.
    """
    CSV_PATH       = r"C:\Users\Admin\Documents\FYP\french dataset\Dataset\french_data.csv"
    STOPWORDS_FILE = r"french_stpwords.txt"
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
# PlotHelper centralises all figure-saving logic in one place.
# Its constructor applies a shared visual style (background, spines, font) to every plot globally.
# save() chains tight_layout → savefig → close → print, eliminating repeated code across 11+ classes.
# safe_name() sanitises label strings (e.g. "Self-Reflection/Growth") into OS-safe filenames.
class PlotHelper:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Apply a shared visual style to every figure created after this point.
        plt.rcParams.update({
            "figure.facecolor" : cfg.BG,
            "axes.facecolor"   : cfg.BG,
            "axes.spines.top"  : False,   # remove top border (cleaner look)
            "axes.spines.right": False,   # remove right border
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
    Uses cfg so the source path and filter value can be changed in one place.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        """Read the CSV (UTF-8 with BOM), filter to French rows, reset index."""
        # encoding="utf-8-sig" strips the BOM byte that Excel often adds.
        df = pd.read_csv(self.cfg.CSV_PATH, encoding="utf-8-sig")
        print(f"[DataLoader] Total rows loaded : {len(df)}")

        # Keep only rows whose language column matches LANGUAGE_VALUE (case-insensitive).
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

# Load the French spaCy model once at module level; disable unused components
# for speed (NER and dependency parser are not needed here).
nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

# Combine three stop-word sources for maximum coverage:
#   1. spaCy's built-in French stop-words
#   2. NLTK's French stop-words
#   3. A small custom set of common French social-media filler words
SPACY_STOPS  = nlp.Defaults.stop_words.copy()
NLTK_STOPS   = set(stopwords.words("french"))
CUSTOM_STOPS = {"bonjour", "salut", "svp", "merci", "plait"}
STOPWORDS    = SPACY_STOPS | NLTK_STOPS | CUSTOM_STOPS


class TextCleaner:
    """
    All text cleaning, tokenisation, lemmatisation, and feature extraction
    needed to convert raw post text into analysis-ready columns.

    Main entry point: fit_transform(df, text_col) — adds these columns:
        cleaned_text      – normalised text (lower, no emoji/URL/mention)
        hashtags          – list of #tags extracted before cleaning
        tokens            – lemmatised, stop-word-free token list
        char_count        – character count of cleaned_text
        word_count        – word count of cleaned_text
        punct_count       – total ?!... marks
        question_count    – count of '?'
        exclamation_count – count of '!'
        ellipsis_count    – count of '...'
        text_nostop       – tokens joined back to a string (for word clouds etc.)
        emoji_count       – emoji count from the *original* text
        emoticon_count    – ASCII emoticon count from the original text
    """

    def __init__(self) -> None:
        self.nlp           = nlp
        self.stopwords_set: Set[str] = STOPWORDS

        # Unicode ranges covering the most common emoji blocks.
        # Used as a fallback after the `emoji` library's own replacement.
        self.emoji_regex = (
            r'[\U0001F600-\U0001F64F]|'   # emoticons
            r'[\U0001F300-\U0001F5FF]|'   # symbols & pictographs
            r'[\U0001F680-\U0001F6FF]|'   # transport & map
            r'[\U00002600-\U000026FF]|'   # misc symbols
            r'[\U00002700-\U000027BF]|'   # dingbats
            r'[\U0001F900-\U0001F9FF]|'   # supplemental symbols
            r'[\U0001FA00-\U0001FA6F]|'   # chess symbols
            r'[\U0001FA70-\U0001FAFF]'    # symbols & pictographs extended-A
        )

        # Common ASCII / text emoticons grouped by emotion.
        self.emoticon_patterns: List[str] = [
            r':\)|:-\)|:\]|=\]|=\)',    # happy
            r':\(|:-\(|:\[|=\[|=\(',    # sad
            r':D|:-D|=D',               # big smile
            r';\)|;-\)',                 # wink
            r':P|:-P|=P',               # tongue out
            r':o|:-o|:O|:-O',           # surprised
            r':/|:-/',                   # skeptical / unsure
            r":'\(",                     # crying
            r'<3',                       # heart
        ]

    # ── Low-level cleaning helpers ────────────────────────────────────────────

    def remove_emojis(self, text: str) -> str:
        """Strip emojis using the `emoji` library first, then regex as a safety net."""
        text_no_emoji = emoji.replace_emoji(text, replace="")
        return re.sub(self.emoji_regex, "", text_no_emoji)

    def replace_urls(self, text: str) -> str:
        """Replace HTTP/HTTPS URLs and bare www. links with the token ' URL '."""
        return re.sub(r'https?://\S+|www\.\S+', ' URL ', text)

    def replace_mentions(self, text: str) -> str:
        """Replace @username mentions with the anonymisation token ' PEOPLE '."""
        return re.sub(r'@\w+', ' PEOPLE ', text)

    def extract_hashtags(self, text: str) -> Tuple[List[str], str]:
        """Pull out all #tags before cleaning. Returns (hashtag_list, text_without_hashtags)."""
        hashtags = re.findall(r'#\w+', text)
        text_without_hashtags = re.sub(r'#\w+', '', text)
        return hashtags, text_without_hashtags

    def standardize_text(self, text: str) -> str:
        """Lowercase the text and collapse newline characters to spaces."""
        return text.lower().replace('\n', ' ').replace('\r', ' ')

    # ── spaCy-based NLP helpers ───────────────────────────────────────────────

    def tokenize(self, cleaned_text: str) -> List[str]:
        """Tokenise with spaCy and keep only alphabetic tokens (no punctuation/numbers)."""
        doc = self.nlp(cleaned_text)
        return [token.text.lower() for token in doc if token.is_alpha]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatise tokens using spaCy. Re-joins first so spaCy uses context."""
        doc = self.nlp(" ".join(tokens))
        return [token.lemma_.lower() for token in doc if token.is_alpha]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out any token present in the combined STOPWORDS set."""
        return [t for t in tokens if t not in self.stopwords_set]

    def clean_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Full cleaning pipeline for a single post:
          1. Lowercase + normalise whitespace
          2. Strip emojis
          3. Replace URLs
          4. Replace @mentions
          5. Extract and remove #hashtags
          6. Remove remaining punctuation (keep !, ?, -, ')
          7. Collapse extra spaces
        Returns (cleaned_text, hashtag_list).
        """
        text = self.standardize_text(text)
        text = self.remove_emojis(text)
        text = self.replace_urls(text)
        text = self.replace_mentions(text)
        hashtags, text = self.extract_hashtags(text)
        text = re.sub(r"[^\w\s!?\-']", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, hashtags

    def preprocess(self, cleaned_text: str) -> List[str]:
        """Convenience: tokenise → lemmatise → remove stop-words in one call."""
        tokens = self.tokenize(cleaned_text)
        lemmas = self.lemmatize(tokens)
        return self.remove_stopwords(lemmas)

    def fit_transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Apply the full cleaning + feature extraction pipeline to every row
        of df[text_col] and return the enriched DataFrame.
        Note: emoji/emoticon counts use the *original* text (before cleaning)
        so that the signal is not lost by the cleaning step.
        """
        df = df.copy()

        # ── Step 1: clean text + extract hashtags ─────────────────────────────
        cleaned_results    = df[text_col].apply(self.clean_text)
        df["cleaned_text"] = cleaned_results.apply(lambda x: x[0])
        df["hashtags"]     = cleaned_results.apply(lambda x: x[1])

        # ── Step 2: tokenise + lemmatise + de-stop ────────────────────────────
        df["tokens"] = df["cleaned_text"].apply(self.preprocess)

        # ── Step 3: surface-level text statistics ─────────────────────────────
        df["char_count"]          = df["cleaned_text"].apply(len)
        df["word_count"]          = df["cleaned_text"].apply(lambda x: len(x.split()))
        df["punct_count"]         = df["cleaned_text"].apply(
            lambda x: x.count('?') + x.count('!') + x.count('...')
        )
        df["question_count"]      = df["cleaned_text"].apply(lambda x: x.count('?'))
        df["exclamation_count"]   = df["cleaned_text"].apply(lambda x: x.count('!'))
        df["ellipsis_count"]      = df["cleaned_text"].apply(lambda x: x.count('...'))

        # ── Step 4: token string for word-cloud / co-occurrence analyses ───────
        df["text_nostop"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

        # ── Step 5: emoji & emoticon counts from the ORIGINAL raw text ─────────
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

# ── Emoji / emoticon sanity check ──────────────────────────────────────────────
# Diagnostic: confirms whether the dataset contains meaningful emoji/emoticon signal.
# Result: only 23/6000+ posts have emojis, 0 have emoticons — dataset is formal/synthetic.
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
    """
    Req 1 — Bar chart + pie chart of label counts.
    Most important sanity check: confirms whether Healthy/Unhealthy classes are balanced.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        counts = df[self.cfg.LABEL_COL].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Label Distribution — mental_state", fontsize=14, fontweight="bold")

        # ── Left: horizontal bar chart with value annotations ──────────────────
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

        # ── Right: pie chart showing class proportions ─────────────────────────
        axes[1].pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
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
    """
    Req 2 — Word-count histograms and character-count boxplots, one subplot per label.
    Detects if one mental-health label correlates with longer/shorter posts.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> None:
        labels = df[self.cfg.LABEL_COL].unique()
        colors = sns.color_palette(self.cfg.PALETTE, len(labels))

        # ── Plot 2a: word-count histogram per label ────────────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
        fig.suptitle("Word Count Distribution by Label", fontsize=14, fontweight="bold")

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["word_count"]
            median = subset.median()
            ax.hist(subset, bins=30, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(median, color="navy", linestyle="--", linewidth=1.5,
                       label=f"Median: {median:.0f}")
            ax.set_title(f"Word Count — {label}")
            ax.set_xlabel("Word Count")
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.tight_layout()
        self.helper.save("02a_wordcount_histogram_by_label.png")

        # ── Plot 2b: character-count boxplot per label ─────────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
        fig.suptitle("Char Count Distribution by Label", fontsize=14, fontweight="bold")

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["char_count"]
            ax.boxplot(
                subset, patch_artist=True,
                boxprops=dict(facecolor=color, color="gray"),
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor=color, alpha=0.4),
            )
            ax.set_title(f"Char Count — {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Char Count")
            ax.set_xticks([])

        plt.tight_layout()
        self.helper.save("02b_charcount_boxplot_by_label.png")


TextLengthAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 3a — Punctuation Analysis (grouped bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
class PunctuationAnalysis:
    """
    Req 3 — Normalised punctuation usage as a grouped bar chart.
    Counts are divided by avg sentence count per label to allow fair comparison.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        punct_cols = ['question_count', 'exclamation_count', 'ellipsis_count']

        # Sentence count per post (at least 1 to avoid division by zero).
        sentence_counts = df['cleaned_text'].apply(
            lambda t: max(len(sent_tokenize(t)), 1)
        )

        # Build per-label normalised punctuation summary.
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
            # Annotate each bar with its exact value.
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
    """
    Req 3b — Same normalised punctuation data as Graph 3a rendered as a table.
    Complements the bar chart for readers who prefer exact numbers.
    """

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

        # One distinct background colour per label row; data cells are near-white.
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

        # Style the header row: bold white text on dark blue background.
        for j in range(len(norm_df.columns) + 1):
            table[0, j].set_text_props(fontweight="bold", color="white")
            table[0, j].set_facecolor("#4C72B0")

        return self.helper.save("03b_punctuation_table.png")


PunctuationTable(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 4 — Word Clouds per Label
# ═══════════════════════════════════════════════════════════════════════════════
class WordCloudAnalysis:
    """
    Req 4 — One word cloud per label using stop-word-free lemmatised tokens.
    Gives an instant visual summary of dominant vocabulary per mental-health label.
    """

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
                collocations=False,   # avoids repeated bigrams in the cloud
            ).generate(text)

            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].axis("off")
            axes[i].set_title(str(label), fontsize=12, fontweight="bold")

        # Hide any leftover empty axes (when labels < rows*cols).
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        path = self.helper.save("04_wordclouds_per_label.png")
        return [path]


WordCloudAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 5 — Co-occurrence Analysis
# ═══════════════════════════════════════════════════════════════════════════════
class CoOccurrenceAnalysis:
    """
    Req 5 — Top word pairs co-occurring in the same post, per label.
    Reveals semantic associations (e.g. "anxiety" + "stress") specific to each label.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def _cooccurrence(self, texts, top_n):
        """Count unique word pairs per post. Uses combinations so (a,b) == (b,a)."""
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
            sns.barplot(x=counts, y=pair_labels, palette="mako", ax=ax)
            ax.set_title(f"Top Word Co-occurrences — {label}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Co-occurrence count")

            fname = f"05_cooccurrence_{self.helper.safe_name(label)}.png"
            paths.append(self.helper.save(fname))

        return paths


CoOccurrenceAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 6 — Common Words, Bigrams & Trigrams
# ═══════════════════════════════════════════════════════════════════════════════
def get_ngrams(words: list, n: int) -> list:
    """Generate n-grams from a flat list of words using a sliding window."""
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


class CommonWordsAnalysis:
    """
    Req 6 — Top N most frequent unigrams, bigrams, and trigrams per label.
    Three separate plots allow comparison of single-word vs multi-word patterns.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def _plot_ngrams(self, df: pd.DataFrame, n: int, title_prefix: str, filename: str) -> str:
        """Build a side-by-side bar chart of the top-N n-grams for each label."""
        labels = df[self.cfg.LABEL_COL].unique()
        cols   = min(2, len(labels))
        rows   = (len(labels) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
        axes = np.array(axes).flatten()
        fig.suptitle(f"Top {self.cfg.TOP_N_WORDS} {title_prefix} per Label",
                     fontsize=14, fontweight="bold")

        for i, label in enumerate(labels):
            words  = " ".join(df[df[self.cfg.LABEL_COL] == label]["text_nostop"]).split()
            ngrams = get_ngrams(words, n)
            freq   = Counter(ngrams).most_common(self.cfg.TOP_N_WORDS)

            if not freq:
                axes[i].axis("off")
                continue

            w, c = zip(*freq)
            sns.barplot(x=list(c), y=list(w), palette="rocket", ax=axes[i])
            axes[i].set_title(str(label), fontsize=11, fontweight="bold")
            axes[i].set_xlabel("Frequency")
            axes[i].set_ylabel("")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        return self.helper.save(filename)

    def analyse(self, df: pd.DataFrame) -> list:
        """Produce three plots: unigrams, bigrams, and trigrams."""
        paths = []
        paths.append(self._plot_ngrams(df, n=1, title_prefix="Common Words",
                                       filename="06_common_words_per_label.png"))
        paths.append(self._plot_ngrams(df, n=2, title_prefix="Bigrams",
                                       filename="06_bigrams_per_label.png"))
        paths.append(self._plot_ngrams(df, n=3, title_prefix="Trigrams",
                                       filename="06_trigrams_per_label.png"))
        return paths


CommonWordsAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 7 — Category Distribution
# ═══════════════════════════════════════════════════════════════════════════════
class CategoryDistribution:
    """
    Bar chart + pie chart of the 5 thematic categories.
    Confirms the French subset is evenly distributed (~20% per category).
    """

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

        axes[1].pie(counts.values, labels=counts.index,
                    autopct="%1.1f%%",
                    colors=sns.color_palette("Set2", len(counts)),
                    startangle=140)
        axes[1].set_title("Proportion per Category")

        return self.helper.save("07_category_distribution.png")


CategoryDistribution(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 8 — Category × Label Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
class CategoryLabelHeatmap:
    """
    Heatmap of category vs mental_state counts.
    Reveals which thematic categories lean more Healthy or Unhealthy.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        cross = pd.crosstab(df["category"], df[self.cfg.LABEL_COL])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.5, ax=ax)
        ax.set_title("Category × Mental State Heatmap",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Mental State")
        ax.set_ylabel("Category")

        return self.helper.save("08_category_label_heatmap.png")


CategoryLabelHeatmap(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 9 — Word Clouds per Category
# ═══════════════════════════════════════════════════════════════════════════════
class CategoryWordCloud:
    """
    One word cloud per thematic category.
    Shows which vocabulary is dominant within each mental-health topic area.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> list:
        categories = df["category"].unique()
        n    = len(categories)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        axes = np.array(axes).flatten()
        fig.suptitle("Word Clouds per Category", fontsize=15, fontweight="bold")

        cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges"]

        for i, cat in enumerate(categories):
            text = " ".join(df[df["category"] == cat]["text_nostop"])
            if not text.strip():
                axes[i].axis("off")
                continue

            wc = WordCloud(width=600, height=350, background_color="white",
                           colormap=cmaps[i % len(cmaps)],
                           max_words=100, collocations=False).generate(text)

            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].axis("off")
            axes[i].set_title(str(cat), fontsize=11, fontweight="bold")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        path = self.helper.save("09_wordclouds_per_category.png")
        return [path]


CategoryWordCloud(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 10 — Emoji & Emoticon Analysis
# ═══════════════════════════════════════════════════════════════════════════════
class EmojiEmoticonAnalysis:
    """
    Req 7 & 8 — Emoji and emoticon usage analysis (2×2 figure).
    NOTE: Dataset is synthetic/formal — only 23/6000+ posts have emojis,
    0 have emoticons. Results included for completeness but carry no signal.
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(
            "Emoji & Emoticon Analysis\n"
            "⚠️ <1% of posts contain emojis — results not statistically meaningful",
            fontsize=13, fontweight="bold"
        )

        # ── Top-left: emoji count distribution ────────────────────────────────
        axes[0, 0].hist(df["emoji_count"], bins=20, color="#F4A460", edgecolor="white")
        axes[0, 0].axvline(df["emoji_count"].mean(), color="red", linestyle="--",
                           label=f"Mean={df['emoji_count'].mean():.2f}")
        axes[0, 0].set_title("Emoji Count Distribution")
        axes[0, 0].set_xlabel("Emojis per text")
        axes[0, 0].legend()

        # ── Top-right: emoticon count distribution ─────────────────────────────
        axes[0, 1].hist(df["emoticon_count"], bins=20, color="#87CEEB", edgecolor="white")
        axes[0, 1].axvline(df["emoticon_count"].mean(), color="red", linestyle="--",
                           label=f"Mean={df['emoticon_count'].mean():.2f}")
        axes[0, 1].set_title("Emoticon Count Distribution")
        axes[0, 1].set_xlabel("Emoticons per text")
        axes[0, 1].legend()

        # ── Bottom-left: average emoji count per label ────────────────────────
        avg_emoji = (df.groupby(self.cfg.LABEL_COL)["emoji_count"]
                       .mean().sort_values(ascending=False))
        sns.barplot(x=avg_emoji.values, y=avg_emoji.index.astype(str),
                    palette=self.cfg.PALETTE, ax=axes[1, 0])
        axes[1, 0].set_title("Avg Emoji Count by Label")
        axes[1, 0].set_xlabel("Avg emoji count")

        # ── Bottom-right: average emoticon count per label ────────────────────
        avg_emot = (df.groupby(self.cfg.LABEL_COL)["emoticon_count"]
                      .mean().sort_values(ascending=False))
        sns.barplot(x=avg_emot.values, y=avg_emot.index.astype(str),
                    palette=self.cfg.PALETTE, ax=axes[1, 1])
        axes[1, 1].set_title("Avg Emoticon Count by Label")
        axes[1, 1].set_xlabel("Avg emoticon count")

        return self.helper.save("10_emoji_emoticon.png")


EmojiEmoticonAnalysis(cfg, helper).analyse(df)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
# Save the cleaned DataFrame so downstream modelling steps can load it
# directly without re-running the full cleaning pipeline.
df.to_csv(f"{cfg.OUTPUT_DIR}/french_cleaned.csv", index=False, encoding="utf-8-sig")

print("=" * 55)
print("  FRENCH EDA PIPELINE — COMPLETE")
print("=" * 55)

saved = [f for f in os.listdir(cfg.OUTPUT_DIR) if f.endswith(".png")]
print(f"\n📊 {len(saved)} plots saved to '{cfg.OUTPUT_DIR}/':")
for f in sorted(saved):
    print(f"   • {f}")

print(f"\n📄 Cleaned CSV : {cfg.OUTPUT_DIR}/french_cleaned.csv")
print("\n✅ All done!")

# ── Actual emojis found in the dataset ────────────────────────────────────────
from collections import Counter

all_emojis = []
for text in df_raw[cfg.TEXT_COL]:  # use df_raw — the original uncleaned text
    all_emojis.extend([item['emoji'] for item in emoji.emoji_list(str(text))])

emoji_counts = Counter(all_emojis)
print(f"\n── Emojis found in dataset ({len(emoji_counts)} unique) ──")
for em, count in emoji_counts.most_common():
    print(f"   {em}  →  {count} times")