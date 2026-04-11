# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import string
import warnings
from collections import Counter
from itertools import combinations

# ── External libraries ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
import spacy
import emoji
from typing import List, Tuple, Set, Dict, Optional  # ✅ added Dict, Optional
from typing import Literal                             # ✅ added Literal
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field

# REMOVED: from your_models import FrenchMentalHealthPost ✅ (defined below)

import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

warnings.filterwarnings("ignore")
print("All imports OK")


## 0. Configurations
class Config:
    CSV_PATH       = r"C:\Users\Admin\Documents\FYP\french dataset\Dataset\french_data.csv"
    STOPWORDS_FILE = r"french_stpwords.txt"
    OUTPUT_DIR     = r"MyResults"

    LANGUAGE_COL   = "language"
    LANGUAGE_VALUE = "French"
    TEXT_COL       = "text"
    LABEL_COL      = "mental_state"  # ✅ used consistently

    BG      = "#F9F9F9"
    DPI     = 150
    PALETTE = "Set2"

    TOP_N_WORDS = 20
    TOP_N_COOC  = 20

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(f"Config ready — output folder: '{cfg.OUTPUT_DIR}'")


## 1. Plot helper
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
        path = os.path.join(self.cfg.OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=self.cfg.DPI, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {filename}")
        return path

    @staticmethod
    def safe_name(text: str) -> str:
        return re.sub(r'[\\/*?"<>|]+', "_", str(text)).strip()

helper = PlotHelper(cfg)
print("PlotHelper ready")


## 2. Data loading
class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.CSV_PATH, encoding="utf-8-sig")   
        print(f"[DataLoader] Total rows loaded : {len(df)}")
        mask = (
            df[self.cfg.LANGUAGE_COL].str.strip().str.lower()
            == self.cfg.LANGUAGE_VALUE.lower()
        )
        df = df[mask].copy().reset_index(drop=True)
        print(f"[DataLoader] French rows kept  : {len(df)}")
        return df


## Models
class FrenchMentalHealthPost(BaseModel):
    text: str = Field(..., description="Social media post content in French")
    word_count: int = Field(..., description="Number of words in the post")
    language: str = Field(default="French", description="Language of the post")
    category: str = Field(..., description="Thematic category e.g. Self-Worth, Anxiety...")
    age: str = Field(..., description="Age group e.g. 'young adult (20-29)', 'elderly (60+)'")
    education_level: str = Field(..., description="Education level")
    formality: str = Field(..., description="Formality level")
    context: str = Field(..., description="Context of the post")
    mental_state: Literal["Healthy", "Unhealthy"] = Field(..., description="Binary mental health label")
    text_length: str = Field(..., description="Descriptive length")
    length_category: str = Field(..., description="Short length label")

    class Config:
        validate_assignment = True
        use_enum_values = True


class FrenchPostAnalysis(BaseModel):
    # --- Base text stats ---
    word_count: int = Field(default=0)
    char_count: int = Field(default=0)
    punct_density: float = Field(default=0.0)

    # --- Punctuation markers ---
    question_count: int = Field(default=0)
    exclamation_count: int = Field(default=0)
    ellipsis_count: int = Field(default=0)
    guillemet_count: int = Field(default=0)

    # --- Emojis / emoticons ---
    emoji_count: int = Field(default=0)
    emojis: List[str] = Field(default_factory=list)
    emoticon_count: int = Field(default=0)
    emoticons: List[str] = Field(default_factory=list)

    # --- French-specific ---
    accented_char_count: int = Field(default=0)
    negation_count: int = Field(default=0)
    hashtags: List[str] = Field(default_factory=list)

    # --- Metadata mirrors ---
    formality: Optional[str] = Field(default=None)
    context: Optional[str] = Field(default=None)
    mental_state: Optional[Literal["Healthy", "Unhealthy"]] = Field(default=None)


## 3. Text cleaning
nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

SPACY_STOPS  = nlp.Defaults.stop_words.copy()
NLTK_STOPS   = set(stopwords.words("french"))
CUSTOM_STOPS = {"bonjour", "salut", "svp", "merci", "plait"}
STOPWORDS    = SPACY_STOPS | NLTK_STOPS | CUSTOM_STOPS


class TextCleaner:
    def __init__(self) -> None:
        self.nlp = nlp
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

        # ✅ added — used in FeatureExtractor.detect_emoticons
        self.emoticon_patterns: List[str] = [
    r':\)|:-\)|:\]|=\]|=\)',        # happy
    r':\(|:-\(|:\[|=\[|=\(',        # sad
    r':D|:-D|=D',                    # big smile
    r';\)|;-\)',                      # wink
    r':P|:-P|=P',                    # tongue
    r':o|:-o|:O|:-O',               # surprised
    r':/|:-/',                        # skeptical
    r":'\(",                          # ✅ fixed — was r":'\\("
    r'<3',                           # heart
]

    def remove_emojis(self, text: str) -> str:
        text_no_emoji = emoji.replace_emoji(text, replace="")
        return re.sub(self.emoji_regex, "", text_no_emoji)

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
        text = re.sub(r"\s+", " ", text).strip()
        return text, hashtags

    def preprocess(self, cleaned_text: str) -> List[str]:
        tokens = self.tokenize(cleaned_text)
        lemmas = self.lemmatize(tokens)
        return self.remove_stopwords(lemmas)

    # ✅ added — called in section 4
    def fit_transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
     df = df.copy()
     cleaned_results = df[text_col].apply(self.clean_text)
     df["cleaned_text"] = cleaned_results.apply(lambda x: x[0])
     df["hashtags"]     = cleaned_results.apply(lambda x: x[1])
     df["tokens"]       = df["cleaned_text"].apply(self.preprocess)
     df["char_count"]   = df["cleaned_text"].apply(len)
     df["word_count"]   = df["cleaned_text"].apply(lambda x: len(x.split()))
     df["punct_count"]       = df["cleaned_text"].apply(      # ✅ added
        lambda x: x.count('?') + x.count('!') + x.count('...')
    )
     df["question_count"]      = df["cleaned_text"].apply(lambda x: x.count('?'))
     df["exclamation_count"]   = df["cleaned_text"].apply(lambda x: x.count('!'))
     df["ellipsis_count"]      = df["cleaned_text"].apply(lambda x: x.count('...'))

       # ✅ needed by WordCloud, CoOccurrence, CommonWords
     df["text_nostop"]       = df["tokens"].apply(lambda tokens: " ".join(tokens))

    # ✅ needed by EmojiEmoticonAnalysis
     df["emoji_count"]       = df[text_col].apply(
                                  lambda x: len(emoji.emoji_list(x)))
     df["emoticon_count"]    = df[text_col].apply(
                                  lambda x: sum(
                                      len(re.findall(p, x, re.IGNORECASE))
                                      for p in self.emoticon_patterns
                                  ))

   
     print(f"[TextCleaner] Cleaned & tokenized {len(df)} rows")
     return df
    


## 4. Main execution
df_raw = DataLoader(cfg).load()
cleaner = TextCleaner()
df = cleaner.fit_transform(df_raw, cfg.TEXT_COL)
df.head(3)


## 4. Feature extraction
class FeatureExtractor:
    def __init__(self, processor: TextCleaner) -> None:
        self.processor = processor

    def count_punctuation(self, text: str) -> Dict[str, int]:
        return {
            'question_count'   : text.count('?'),
            'exclamation_count': text.count('!'),
            'ellipsis_count'   : text.count('...'),
        }

    def detect_emojis_combined(self, text: str) -> Tuple[List[str], int]:
        found_emojis = [item['emoji'] for item in emoji.emoji_list(text)]
        standardized = [emoji.demojize(e) for e in found_emojis]
        regex_matches = re.findall(self.processor.emoji_regex, text)
        for match in regex_matches:
            name = emoji.demojize(match)
            if name not in standardized:
                standardized.append(name)
        return standardized, len(standardized)

    def detect_emoticons(self, text: str) -> Tuple[List[str], int]:
        emoticons: List[str] = []
        for pattern in self.processor.emoticon_patterns:  # ✅ now exists
            matches = re.findall(pattern, text, re.IGNORECASE)
            emoticons.extend(matches)
        return emoticons, len(emoticons)

    def extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if len(tokens) < n:
            return []
        ngram_list = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return [ng for ng in ngram_list if all(word.isalpha() for word in ng.split())]

    def compute_post_features_from_doc(
        self, original_text: str, cleaned_text: str, doc
    ) -> FrenchPostAnalysis:  # ✅ was PostAnalysis, now FrenchPostAnalysis
        word_count   = sum(1 for token in doc if token.is_alpha)
        char_count   = len(cleaned_text)
        punct_counts = self.count_punctuation(cleaned_text)
        total_punct  = sum(punct_counts.values())
        punct_density = total_punct / word_count if word_count > 0 else 0.0
        emojis, emoji_count       = self.detect_emojis_combined(original_text)
        emoticons, emoticon_count = self.detect_emoticons(original_text)

        return FrenchPostAnalysis(  # ✅ was PostAnalysis
            word_count=word_count, char_count=char_count,
            punct_density=punct_density,
            question_count=punct_counts.get('question_count', 0),
            exclamation_count=punct_counts.get('exclamation_count', 0),
            ellipsis_count=punct_counts.get('ellipsis_count', 0),
            emoji_count=emoji_count, emojis=emojis,
            emoticon_count=emoticon_count, emoticons=emoticons,
            hashtags=[]
        )


## 5. Text analyzer
class TextAnalyzer:
    def __init__(self, processor: TextCleaner, feature_extractor: FeatureExtractor) -> None:
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.labels: List[str] = []
        self.label_tokens: Dict[str, List[List[str]]] = {}

    def compute_vocabulary_diversity(self, tokens: List[str]) -> float:
        if len(tokens) == 0:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def compute_normalized_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        punct_cols = ['question_count', 'exclamation_count', 'ellipsis_count']
        sentence_counts = df['text'].apply(
            lambda text: max(len(sent_tokenize(text)), 1)
        )
        summary_data = []
        for label in df[cfg.LABEL_COL].unique():  # ✅ was df['status']
            label_data   = df[df[cfg.LABEL_COL] == label]
            avg_sent_len = sentence_counts[label_data.index].mean()
            normalized_punct = {}
            for col in punct_cols:
                avg_count = label_data[col].mean()
                normalized_punct[col.replace('_count', '').title()] = (
                    avg_count / avg_sent_len if avg_sent_len > 0 else 0
                )
            normalized_punct['Label'] = label
            summary_data.append(normalized_punct)
        return pd.DataFrame(summary_data).set_index('Label')

    def compute_label_top_ngrams(
        self, df: pd.DataFrame, text_column: str,
        label_column: str, n: int, top_k: int = 10
    ) -> Dict[str, List[Tuple[str, int]]]:
        label_ngrams: Dict[str, List[Tuple[str, int]]] = {}
        for label in df[label_column].unique():
            label_mask    = df[label_column] == label
            ngram_counter: Counter = Counter()
            if 'tokens' in df.columns:
                for tokens in df[label_mask]['tokens']:
                    if isinstance(tokens, list):
                        ngram_counter.update(self.feature_extractor.extract_ngrams(tokens, n))
            else:
                for text in df[label_mask][text_column]:
                    tokens = self.processor.preprocess(text)
                    ngram_counter.update(self.feature_extractor.extract_ngrams(tokens, n))
            label_ngrams[label] = ngram_counter.most_common(top_k)
        return label_ngrams

    def compute_shared_vocabulary(self, df: pd.DataFrame) -> Dict[str, Set[str]]:
        label_vocab: Dict[str, Set[str]] = {}
        for label in df[cfg.LABEL_COL].unique():  # ✅ was df['status']
            label_mask = df[cfg.LABEL_COL] == label
            vocab: Set[str] = set()
            if 'tokens' in df.columns:
                for tokens in df[label_mask]['tokens']:
                    if isinstance(tokens, list):
                        vocab.update(t for t in tokens if t and t.isalpha())
            else:
                text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
                for text in df[label_mask][text_col]:
                    if not isinstance(text, str):
                        continue
                    tokens = self.processor.preprocess(text)
                    vocab.update(t for t in tokens if t and t.isalpha())
            label_vocab[label] = vocab
        return label_vocab

# Graph1 Label Distribution
class LabelDistribution:
    """Req 1 — Bar chart + pie chart of label counts."""

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        counts = df[self.cfg.LABEL_COL].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Label Distribution — mental_state", fontsize=14, fontweight="bold")

        # Bar chart
        sns.barplot(x=counts.values, y=counts.index.astype(str),
                    palette=self.cfg.PALETTE, ax=axes[0])
        axes[0].set_xlabel("Count")
        axes[0].set_title("Count per Label")
        for bar, val in zip(axes[0].patches, counts.values):
            axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                         str(val), va="center", fontsize=9)

        # Pie chart
        axes[1].pie(counts.values, labels=counts.index,
                    autopct="%1.1f%%",
                    colors=sns.color_palette(self.cfg.PALETTE, len(counts)),
                    startangle=140)
        axes[1].set_title("Proportion per Label")

        return self.helper.save("1_label_distribution.png")

LabelDistribution(cfg, helper).analyse(df)

#2 Graph 2 Text length analysis 
class TextLengthAnalysis:
    """Histogram of word count + boxplot of char count, one per label."""

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> None:
        labels = df[self.cfg.LABEL_COL].unique()
        colors = sns.color_palette(self.cfg.PALETTE, len(labels))

        # ── Plot 1: One histogram per label ──────────────────────────────────
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

        # ── Plot 2: One boxplot per label ─────────────────────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
        fig.suptitle("Char Count Distribution by Label", fontsize=14, fontweight="bold")

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["char_count"]

            ax.boxplot(subset, patch_artist=True,
                       boxprops=dict(facecolor=color, color="gray"),
                       medianprops=dict(color="black", linewidth=2),
                       flierprops=dict(marker='o', markersize=3,
                                       markerfacecolor=color, alpha=0.4))
            ax.set_title(f"Char Count — {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Char Count")
            ax.set_xticks([])

        plt.tight_layout()
        self.helper.save("02b_charcount_boxplot_by_label.png")

TextLengthAnalysis(cfg, helper).analyse(df)

class PunctuationAnalysis:
    """Normalized punctuation usage as a grouped bar chart."""

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        # ── Normalize punctuation counts by sentence count ────────────────────
        punct_cols = ['question_count', 'exclamation_count', 'ellipsis_count']
        sentence_counts = df['cleaned_text'].apply(
            lambda t: max(len(sent_tokenize(t)), 1)
        )

        summary = []
        for label in df[self.cfg.LABEL_COL].unique():
            mask         = df[self.cfg.LABEL_COL] == label
            avg_sents    = sentence_counts[mask].mean()
            row          = {"Label": label}
            for col in punct_cols:
                avg_count    = df[mask][col].mean()
                row[col]     = avg_count / avg_sents if avg_sents > 0 else 0
            summary.append(row)

        norm_df = pd.DataFrame(summary).set_index("Label")
        norm_df.columns = ["Question", "Exclamation", "Ellipsis"]  # clean names

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle("Punctuation Usage Normalized by Sentence Length",
                     fontsize=14, fontweight="bold")

        x      = np.arange(len(norm_df.columns))   # 3 punct types on x-axis
        labels = norm_df.index.tolist()             # Healthy / Unhealthy
        n      = len(labels)
        width  = 0.35
        colors = sns.color_palette(self.cfg.PALETTE, n)

        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(x + offset, norm_df.loc[label],
                            width=width, label=label,
                            color=color, edgecolor="white")
            # value labels on top of each bar
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{bar.get_height():.4f}",
                        ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(norm_df.columns, fontsize=11)
        ax.set_ylabel("Avg count per sentence")
        ax.set_xlabel("Punctuation type")
        ax.legend(title="Label")
        ax.set_ylim(0, norm_df.values.max() * 1.25)  # headroom for value labels

        return self.helper.save("03_punctuation_normalized.png")

PunctuationAnalysis(cfg, helper).analyse(df)

class PunctuationTable:
    """Normalized punctuation table rendered as a styled figure."""

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        # ── Compute normalized values ─────────────────────────────────────────
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
                row[col]  = round(avg_count / avg_sents if avg_sents > 0 else 0, 4)
            summary.append(row)

        norm_df = pd.DataFrame(summary).set_index("Label")
        norm_df.columns = ["Question", "Exclamation", "Ellipsis"]

        # ── Draw table as matplotlib figure ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 2 + len(norm_df) * 0.6))
        ax.axis("off")
        fig.suptitle("Punctuation Usage Normalized by Sentence Length",
                     fontsize=13, fontweight="bold", y=1.02)

        colors_rows = sns.color_palette(self.cfg.PALETTE, len(norm_df))
        row_colors  = [[c] + ["#f9f9f9"] * len(norm_df.columns)
                       for c in colors_rows]

        table = ax.table(
            cellText   = norm_df.reset_index().values,
            colLabels  = ["Label"] + list(norm_df.columns),
            cellLoc    = "center",
            loc        = "center",
            cellColours= row_colors,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.3, 2)

        # Bold header
        for j in range(len(norm_df.columns) + 1):
            table[0, j].set_text_props(fontweight="bold", color="white")
            table[0, j].set_facecolor("#4C72B0")

        return self.helper.save("03b_punctuation_table.png")

PunctuationTable(cfg, helper).analyse(df)

class WordCloudAnalysis:
    """Req 4 — One word cloud per label (using text_nostop)."""

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

class CoOccurrenceAnalysis:
    """Req 5 — Top word pairs that appear together per label (horizontal bar chart)."""

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
            sns.barplot(x=counts, y=pair_labels, palette="mako", ax=ax)
            ax.set_title(f"Top Word Co-occurrences — {label}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Co-occurrence count")

            fname = f"05_cooccurrence_{self.helper.safe_name(label)}.png"
            paths.append(self.helper.save(fname))

        return paths

CoOccurrenceAnalysis(cfg, helper).analyse(df)

from collections import Counter
from itertools import islice

def get_ngrams(words: list, n: int) -> list:
    """Generate n-grams from a list of words."""
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


class CommonWordsAnalysis:
    """Req 6 — Top N most frequent words, bigrams, and trigrams per label."""

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def _plot_ngrams(
        self,
        df: pd.DataFrame,
        n: int,
        title_prefix: str,
        filename: str,
    ) -> str:
        labels = df[self.cfg.LABEL_COL].unique()
        cols   = min(2, len(labels))
        rows   = (len(labels) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
        axes = np.array(axes).flatten()
        fig.suptitle(
            f"Top {self.cfg.TOP_N_WORDS} {title_prefix} per Label",
            fontsize=14, fontweight="bold",
        )

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
        paths = []

        # Unigrams (original behaviour)
        paths.append(self._plot_ngrams(
            df, n=1,
            title_prefix="Common Words",
            filename="06_common_words_per_label.png",
        ))

        # Bigrams
        paths.append(self._plot_ngrams(
            df, n=2,
            title_prefix="Bigrams",
            filename="06_bigrams_per_label.png",
        ))

        # Trigrams
        paths.append(self._plot_ngrams(
            df, n=3,
            title_prefix="Trigrams",
            filename="06_trigrams_per_label.png",
        ))

        return paths


CommonWordsAnalysis(cfg, helper).analyse(df)

class EmojiEmoticonAnalysis:
    """
    Req 7 & 8 — Emoji and emoticon usage analysis.
    Produces:
      - Bar chart: avg emoji count per label
      - Bar chart: avg emoticon count per label
      - Histogram: distribution of emoji counts
      - Histogram: distribution of emoticon counts
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle("Emoji & Emoticon Analysis", fontsize=14, fontweight="bold")

        # ── Top-left: emoji count distribution (histogram) ────────────────────
        axes[0, 0].hist(df["emoji_count"], bins=20, color="#F4A460", edgecolor="white")
        axes[0, 0].axvline(df["emoji_count"].mean(), color="red", linestyle="--",
                           label=f"Mean={df['emoji_count'].mean():.2f}")
        axes[0, 0].set_title("Emoji Count Distribution")
        axes[0, 0].set_xlabel("Emojis per text")
        axes[0, 0].legend()

        # ── Top-right: emoticon count distribution (histogram) ────────────────
        axes[0, 1].hist(df["emoticon_count"], bins=20, color="#87CEEB", edgecolor="white")
        axes[0, 1].axvline(df["emoticon_count"].mean(), color="red", linestyle="--",
                           label=f"Mean={df['emoticon_count'].mean():.2f}")
        axes[0, 1].set_title("Emoticon Count Distribution")
        axes[0, 1].set_xlabel("Emoticons per text")
        axes[0, 1].legend()

        # ── Bottom-left: avg emoji count per label (bar chart) ────────────────
        avg_emoji = (df.groupby(self.cfg.LABEL_COL)["emoji_count"]
                       .mean().sort_values(ascending=False))
        sns.barplot(x=avg_emoji.values, y=avg_emoji.index.astype(str),
                    palette=self.cfg.PALETTE, ax=axes[1, 0])
        axes[1, 0].set_title("Avg Emoji Count by Label")
        axes[1, 0].set_xlabel("Avg emoji count")

        # ── Bottom-right: avg emoticon count per label (bar chart) ────────────
        avg_emot = (df.groupby(self.cfg.LABEL_COL)["emoticon_count"]
                      .mean().sort_values(ascending=False))
        sns.barplot(x=avg_emot.values, y=avg_emot.index.astype(str),
                    palette=self.cfg.PALETTE, ax=axes[1, 1])
        axes[1, 1].set_title("Avg Emoticon Count by Label")
        axes[1, 1].set_xlabel("Avg emoticon count")

        return self.helper.save("07_08_emoji_emoticon.png")

EmojiEmoticonAnalysis(cfg, helper).analyse(df)


# Check a few Healthy posts that contain "dépression"
mask = (
    df[cfg.LABEL_COL].str.lower() == "healthy"
) & (
    df["text"].str.lower().str.contains("dépression")
)
print(df[mask]["text"].head(10).tolist())