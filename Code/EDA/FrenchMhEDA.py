"""
mental_health_EDA.py
====================
Exploratory Data Analysis (EDA) pipeline for a French mental-health social-media dataset.

PIPELINE OVERVIEW
-----------------
  0. Config         – single place for all settings (paths, column names, visual style)
  1. PlotHelper     – handles saving figures and sanitising filenames
  2. DataLoader     – loads the CSV, filters to French rows, removes duplicates
  3. TextCleaner    – cleans raw text, tokenises, lemmatises, extracts features
  4. Analysis classes – one class per chart group (Graphs 1–10)

COLUMNS PRODUCED BY TextCleaner.fit_transform()
------------------------------------------------
  cleaned_text      – full sentence after noise removal (keeps !, ?, -, ')
  hashtags          – list of #tags extracted before cleaning
  tokens            – lemmatised, stop-word-free list of meaningful words
  char_count        – number of characters in cleaned_text
  text_length       – number of words (space-split) in cleaned_text
  punct_count       – total count of ?, !, and ... in cleaned_text
  question_count    – count of ? marks
  exclamation_count – count of ! marks
  ellipsis_count    – count of ... sequences
  text_nostop       – tokens joined back into a single string (for NLP graphs)
  emoji_count       – emoji count from the ORIGINAL raw text
  emoticon_count    – ASCII emoticon count from the ORIGINAL raw text

WHERE EACH COLUMN IS USED
--------------------------
  cleaned_text  → char_count, text_length, all punct counts, sent_tokenize (Graph 3)
  tokens        → text_nostop; also filtered with WORDCLOUD_STOPWORDS for Graph 4
  text_nostop   → Graph 5 (co-occurrence), Graph 6 (n-grams)
  char_count    → Graph 2c (char count boxplot)
  text_length   → Graph 2a (histogram), Graph 2b (boxplot)
  punct counts  → Graph 3a (bar chart), Graph 3b (table)
  emoji_count   → Graph 10
  emoticon_count→ Graph 10

STOPWORD SETS — TWO SEPARATE LISTS
------------------------------------
  STOPWORDS           – used during text cleaning (tokenise → lemmatise → remove_stopwords)
                        keeps negations and intensity modifiers (they matter for meaning)
  WORDCLOUD_STOPWORDS – used ONLY in Graph 4 (word clouds)
                        extends STOPWORDS to also remove negations and intensity modifiers
                        because these words dominate word clouds and hide content words
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Standard library ──────────────────────────────────────────────────────────
import os           # file/folder operations (makedirs, listdir, path.join)
import re           # regular expressions for text cleaning
import string       # string constants (not directly used but kept for reference)
import warnings     # suppress non-critical library warnings
from collections import Counter       # fast frequency counting (emojis, co-occurrence)
from itertools import combinations    # generate word pairs for co-occurrence

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np                    # numerical arrays (subplot layout, bar offsets)
import pandas as pd                   # DataFrame operations (load CSV, group, crosstab)
import matplotlib.pyplot as plt       # base plotting engine
import seaborn as sns                 # high-level statistical plots (barplot, heatmap)
from wordcloud import WordCloud       # word cloud images (Graph 4)
import spacy                          # French NLP: tokenisation + lemmatisation
import emoji                          # emoji detection and removal

# ── Typing helpers ────────────────────────────────────────────────────────────
from typing import List, Tuple, Set, Dict, Optional

# ── NLTK sentence tokeniser (used in punctuation normalisation, Graph 3) ──────
from nltk.tokenize import sent_tokenize

# ── NLTK resource downloads ───────────────────────────────────────────────────
# 'punkt' and 'punkt_tab' are required by sent_tokenize.
# 'wordnet' is downloaded as a precaution (not actively used here).
# All downloads are silent; failures are non-fatal so the pipeline still runs.
import nltk
try:
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet',   quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

warnings.filterwarnings("ignore")   # hide deprecation / future warnings from libraries
print("All imports OK")


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    """
    Central store for every tuneable constant in the pipeline.

    WHY A CLASS INSTEAD OF GLOBALS?
    Passing `cfg` into every class means:
      – One change here propagates everywhere automatically.
      – No hunting for magic strings scattered across 500+ lines.
      – Easy to swap paths or column names without touching analysis code.
    """

    # ── Input / output paths ──────────────────────────────────────────────────
    CSV_PATH   = r"C:\Users\Admin\Documents\FYP\french dataset\Dataset\french_data.csv"
    OUTPUT_DIR = r"MyResults"          # all PNG plots + cleaned CSV saved here

    # ── Column names in the raw CSV ───────────────────────────────────────────
    LANGUAGE_COL   = "language"        # column that identifies the post language
    LANGUAGE_VALUE = "French"          # value to keep (case-insensitive filter)
    TEXT_COL       = "text"            # column containing the raw post text
    LABEL_COL      = "mental_state"    # target label (e.g. Healthy / Unhealthy)

    # ── Visual style (applied globally via PlotHelper) ────────────────────────
    BG      = "#F9F9F9"   # off-white background for all figures
    DPI     = 150          # resolution for saved PNGs
    PALETTE = "Set2"       # seaborn colour palette (colour-blind friendly)

    # ── Analysis limits ───────────────────────────────────────────────────────
    TOP_N_WORDS = 20   # how many top words/n-grams to show in Graph 6
    TOP_N_COOC  = 20   # how many top co-occurring pairs to show in Graph 5


cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)   # create output folder if it doesn't exist
print(f"Config ready — output folder: '{cfg.OUTPUT_DIR}'")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLOT HELPER
# ══════════════════════════════════════════════════════════════════════════════
class PlotHelper:
    """
    Centralises all figure-saving logic so it isn't repeated across 10+ classes.

    WHAT IT DOES
    ─────────────
    __init__  → applies a shared visual style (background, spines, font) to
                every matplotlib figure created after this point.
    save()    → chains tight_layout → savefig → close → prints confirmation.
    safe_name → converts a label string (e.g. "Self-Reflection/Growth") into
                an OS-safe filename by replacing illegal characters.

    USAGE
      helper.save("01_label_distribution.png")
      helper.safe_name("Anxiety/Depression")  →  "Anxiety_Depression"
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Apply shared style GLOBALLY — every figure created after this inherits it.
        plt.rcParams.update({
            "figure.facecolor" : cfg.BG,    # figure background colour
            "axes.facecolor"   : cfg.BG,    # axes (plot area) background colour
            "axes.spines.top"  : False,     # hide top border line (cleaner look)
            "axes.spines.right": False,     # hide right border line
            "font.size"        : 11,        # default font size for all text
        })

    def save(self, filename: str) -> str:
        """
        Save the CURRENT active matplotlib figure to OUTPUT_DIR.
        Steps: tight_layout (removes clipping) → savefig → close (free memory).
        Returns the full path so callers can log or chain it.
        """
        path = os.path.join(self.cfg.OUTPUT_DIR, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=self.cfg.DPI, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {filename}")
        return path

    @staticmethod
    def safe_name(text: str) -> str:
        """
        Replace characters that are illegal in Windows/Linux filenames
        (\\, /, *, ?, ", <, >, |) with underscores.
        E.g. "Self-Reflection/Growth" → "Self-Reflection_Growth"
        """
        return re.sub(r'[\\/*?"<>|]+', "_", str(text)).strip()


helper = PlotHelper(cfg)
print("PlotHelper ready")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
class DataLoader:
    """
    Loads the raw CSV and returns only the French-language rows,
    with duplicate posts removed.

    STEPS INSIDE load()
    ───────────────────
    1. Read CSV with UTF-8-BOM encoding (strips the BOM byte Excel often adds).
    2. Filter rows where LANGUAGE_COL == "French" (case-insensitive).
    3. Remove duplicate posts (same text, regardless of label).
    4. Reset the index so row numbers are 0…N-1.

    OUTPUT
    ──────
    Returns a clean pd.DataFrame with only French, unique posts.
    This is stored as `df_raw` and passed to TextCleaner.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        # Step 1 — load the full CSV
        df = pd.read_csv(self.cfg.CSV_PATH, encoding="utf-8-sig")
        print(f"[DataLoader] Total rows loaded : {len(df)}")

        # Step 2 — keep only French rows (case-insensitive comparison)
        mask = (
            df[self.cfg.LANGUAGE_COL].str.strip().str.lower()
            == self.cfg.LANGUAGE_VALUE.lower()
        )
        df = df[mask].copy()

        # Step 3 — remove duplicate posts based on the text column only.
        # Using subset=[TEXT_COL] means two posts with the same text but
        # different labels are still treated as duplicates — only the FIRST
        # occurrence is kept.
        before = len(df)
        df = df.drop_duplicates(subset=self.cfg.TEXT_COL).reset_index(drop=True)

        print(f"[DataLoader] French rows kept  : {before}")
        print(f"[DataLoader] After dedup       : {len(df)} ({before - len(df)} duplicates removed)")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEXT CLEANER — stopword and vocabulary definitions
# ══════════════════════════════════════════════════════════════════════════════

# Load the French spaCy model ONCE at module level (expensive operation).
# Disabling "ner" (named-entity recognition) and "parser" (dependency tree)
# speeds up tokenisation and lemmatisation since we don't need those components.
nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])

# ── PRONOUNS — removed because they carry no mental-health signal ──────────
PRONOUNS = {
    "je", "j", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "me", "moi", "te", "toi", "se",
    # Possessive determiners (mon, ma, mes …)
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs"
}

# ── EXTRA_REMOVE — grammar/structure words with no content value ────────────
EXTRA_REMOVE = {
    "le", "la", "les",                                           # definite articles
    "un", "une", "des",                                          # indefinite articles
    "du", "au", "aux",                                           # contracted articles
    "de", "à", "en", "dans", "sur", "avec", "pour", "par",      # prepositions
    "sans", "chez",
    "et", "ou", "mais", "donc", "or", "ni", "car",              # coordinating conjunctions
    "que", "qui", "quand", "lorsque", "comme", "puisque",        # subordinating conjunctions
    "quoique", "quoi", "si", "afin", "bien", "pendant",
    "avant", "après", "depuis", "jusqu", "malgré",
    "chaque", "tous", "toutes", "tout", "toute",                 # quantifiers
    "ce", "cet", "cette", "ces",     # demonstrative determiners
    "toujours" , "parfois"         }             #  frequency adverbs

# ── NOISE — single-character and truncated fragments that survive tokenisation
# but are too short or ambiguous to carry meaning.
# e.g. "j" may come from "j'ai" after spaCy splits the contraction —
# spaCy's lemmatiser needs context to correctly resolve "j" → "je",
# and even if it does, "je" is already in PRONOUNS and would be removed anyway.
# Keeping "j" in NOISE ensures it is caught regardless of whether
# lemmatisation resolves it correctly or not — it is a safety net.
NOISE = {"j", "m", "n", "s", "t", "quelqu", "aujourd", "hui", "pa"}

# ── REMOVE_VERBS — all conjugations of être (to be) and avoir (to have) ─────
# These auxiliary verbs appear in virtually every sentence and add no signal.
REMOVE_VERBS = {
    "être", "avoir",
    # être — present
    "suis", "es", "est", "sommes", "êtes", "sont",
    # être — imperfect
    "étais", "était", "étions", "étiez", "étaient",
    # être — future
    "serai", "seras", "sera", "serons", "serez", "seront",
    # être — conditional
    "serais", "serait", "serions", "seriez", "seraient",
    # être — subjunctive
    "sois", "soit", "soyons", "soyez", "soient",
    # avoir — present
    "ai", "as", "a", "avons", "avez", "ont",
    # avoir — imperfect
    "avais", "avait", "avions", "aviez", "avaient",
    # avoir — future
    "aurai", "auras", "aura", "aurons", "aurez", "auront",
    # avoir — conditional
    "aurais", "aurait", "aurions", "auriez", "auraient",
    # avoir — subjunctive
    "aie", "aies", "ait", "ayons", "ayez", "aient",
    # past participles (compound tenses)
    "été", "eu",
}

# ── KEEP_WORDS — override: these words must NEVER be removed from STOPWORDS ──
# Negation words (ne, pas, jamais …) and intensity modifiers (trop, tellement …)
# carry critical mental-health signal:
#   – negations flip meaning: "pas de douleur" ≠ "douleur"
#   – intensity modifiers signal severity: "trop triste" ≠ "triste"
# Domain-specific terms are also protected for obvious relevance.
# NOTE: These words ARE removed in WORDCLOUD_STOPWORDS below because they
# dominate word clouds visually without adding interpretive value there.
KEEP_WORDS = {
    "ne", "pas", "rien", "personne", "jamais",                   # negations
    "plus", "toujours", "parfois", "tellement", "trop",          # intensity modifiers
    "dépression", "pensées", "vide", "douleur", "désespoir",     # domain terms
    "espoir", "suicidaires", "lumière", "obscurité", "âme",
    "résilience", "guérison"
}

# ── STOPWORDS — PRIMARY list used during text cleaning ────────────────────────
# Union of all removal sets, then subtract KEEP_WORDS so protected terms survive.
# Used in: TextCleaner.remove_stopwords() → applied to every post during cleaning.
# NOT used in word clouds 
STOPWORDS = (PRONOUNS | EXTRA_REMOVE | NOISE | REMOVE_VERBS) - KEEP_WORDS

# ══════════════════════════════════════════════════════════════════════════════
# WORDCLOUD-SPECIFIC STOPWORDS
# ══════════════════════════════════════════════════════════════════════════════
# Used ONLY in Graph 4 (WordCloudAnalysis). NOT used anywhere else in the pipeline.
#
# WHY A SEPARATE LIST FOR WORD CLOUDS?
# ──────────────────────────────────────
# During text cleaning, negations (pas, jamais, ne …) and intensity modifiers
# (trop, tellement, toujours …) are KEPT in STOPWORDS via KEEP_WORDS because
# they carry critical semantic signal — e.g. "ne pas souffrir" means the
# opposite of "souffrir", and removing "pas" would destroy that meaning.
#
# However, in a word cloud these same words appear so frequently across ALL
# posts that they visually dominate the cloud and hide the actual content words
# (douleur, espoir, anxiété …) that differ meaningfully between labels.
#
# Solution: extend STOPWORDS with negations and intensity modifiers for word
# clouds only. This makes the visualisation far more informative without
# affecting cleaning, tokenisation, n-grams, or co-occurrence analysis.
#
# This list does NOT replace STOPWORDS — it is applied only inside
# WordCloudAnalysis.analyse() by filtering df["tokens"] directly.
WORDCLOUD_STOPWORDS = STOPWORDS | {
    # Negations — kept in STOPWORDS but removed here for visual clarity
    "ne", "pas", "rien", "personne", "jamais",
    # Intensity / frequency modifiers — dominate clouds without adding label insight
    "très", "trop", "toujours", "parfois", "tellement",
    "plus", "bien", "vraiment", "encore", "déjà",
    "assez", "peu", "beaucoup", "moins", "autant",
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEXT CLEANER — main class
# ══════════════════════════════════════════════════════════════════════════════
class TextCleaner:
    """
    Converts raw post text into analysis-ready columns via two parallel branches:

    BRANCH 1 — Statistical (uses cleaned_text directly)
    ────────────────────────────────────────────────────
    cleaned_text keeps punctuation (!, ?) so counts can be extracted BEFORE
    lemmatisation destroys them. Columns produced:
      char_count, text_length, question_count, exclamation_count,
      ellipsis_count, punct_count

    BRANCH 2 — Linguistic (uses tokens → text_nostop)
    ──────────────────────────────────────────────────
    cleaned_text is passed through tokenise → lemmatise → remove_stopwords.
    is_alpha inside tokenise naturally discards !, ?, -, ' so they never
    appear in word clouds or n-gram charts. Columns produced:
      tokens, text_nostop

    NOTE ON - AND '
    ───────────────
    The cleaning regex retains - and ' to preserve French compound words
    (bien-être) and contractions (j'ai) in cleaned_text. However, spaCy's
    tokeniser and is_alpha already handle these correctly, so the retention
    is redundant for the linguistic branch — it has zero impact on results.
    """

    def __init__(self) -> None:
        self.nlp            = nlp
        self.stopwords_set: Set[str] = STOPWORDS

        # ── Emoji Unicode ranges (safety-net regex after emoji library) ───────
        # The emoji library handles most cases; this regex catches any stragglers
        # in known Unicode blocks that slip through.
        self.emoji_regex = (
            r'[\U0001F600-\U0001F64F]|'   # emoticons (😀 … 🙏)
            r'[\U0001F300-\U0001F5FF]|'   # symbols & pictographs (🌅 … 🗿)
            r'[\U0001F680-\U0001F6FF]|'   # transport & map (🚀 … 🛿)
            r'[\U00002600-\U000026FF]|'   # misc symbols (☀ … ⛿)
            r'[\U00002700-\U000027BF]|'   # dingbats (✀ … ➿)
            r'[\U0001F900-\U0001F9FF]|'   # supplemental symbols (🤐 … 🧿)
            r'[\U0001FA00-\U0001FA6F]|'   # chess symbols
            r'[\U0001FA70-\U0001FAFF]'    # symbols & pictographs extended-A
        )

        # ── ASCII / text emoticons grouped by emotion ─────────────────────────
        # Used ONLY for emoticon_count — counted from the ORIGINAL raw text.
        self.emoticon_patterns: List[str] = [
            r':\)|:-\)|:\]|=\]|=\)',    # happy   :)  :-)
            r':\(|:-\(|:\[|=\[|=\(',    # sad     :(  :-(
            r':D|:-D|=D',               # big smile
            r';\)|;-\)',                 # wink    ;)
            r':P|:-P|=P',               # tongue out
            r':o|:-o|:O|:-O',           # surprised
            r':/|:-/',                   # skeptical
            r":'\(",                     # crying  :'(
            r'<3',                       # heart
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # LOW-LEVEL CLEANING HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def remove_emojis(self, text: str) -> str:
        """
        Two-pass emoji removal:
          Pass 1 — emoji.replace_emoji() handles known Unicode sequences
                   including ZWJ sequences, flags, and variation selectors.
          Pass 2 — regex catches any remaining characters in known emoji blocks.
        Called TWICE in clean_text: before and after the punctuation regex,
        to ensure nothing slips through.
        """
        text = emoji.replace_emoji(text, replace="")
        text = re.sub(self.emoji_regex, "", text)
        return text

    def replace_urls(self, text: str) -> str:
        """
        Replace http/https URLs and bare www. links with the anonymisation
        token ' URL '. This reduces noise while preserving sentence structure
        — the token signals that a link was present without keeping the URL itself.
        """
        return re.sub(r'https?://\S+|www\.\S+', ' URL ', text)

    def replace_mentions(self, text: str) -> str:
        """
        Replace @username mentions with the anonymisation token ' PEOPLE '.
        This ensures user privacy and removes noise.
        NOTE: The dataset contains no @mentions — this step produces zero
        replacements but is kept for robustness and pipeline completeness.
        The count is confirmed in the diagnostic printout inside fit_transform().
        """
        return re.sub(r'@\w+', ' PEOPLE ', text)

    def extract_hashtags(self, text: str) -> Tuple[List[str], str]:
        """
        Extract #hashtags BEFORE cleaning so they are not lost.
        Returns (list_of_hashtags, text_without_hashtags).
        The hashtag list is stored in df["hashtags"] for potential downstream
        analysis (e.g. topic modelling, hashtag frequency analysis).
        Hashtags are removed from the main text to avoid polluting token counts.
        """
        hashtags = re.findall(r'#\w+', text)
        text_without_hashtags = re.sub(r'#\w+', '', text)
        return hashtags, text_without_hashtags

    def standardize_text(self, text: str) -> str:
        """
        Lowercase the text and collapse \\n / \\r to spaces.
        Must run FIRST so all subsequent steps work on clean, uniform text.
        """
        return text.lower().replace('\n', ' ').replace('\r', ' ')

    # ══════════════════════════════════════════════════════════════════════════
    # SPACY-BASED NLP HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def tokenize(self, cleaned_text: str) -> List[str]:
        """
        Tokenise with spaCy and keep ONLY alphabetic tokens (token.is_alpha).
        This naturally drops:  !, ?, -, ', ..., numbers, URLs, punctuation.
        Result: a list of lowercase word strings.
        Note: is_alpha is why - and ' retained in cleaned_text don't matter here.
        """
        doc = self.nlp(cleaned_text)
        return [token.text.lower() for token in doc if token.is_alpha]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatise tokens using spaCy's French model.
        Tokens are re-joined into a sentence first so spaCy uses context
        (e.g. "tristes" → "triste"; "pensées" → "pensée").
        Result: list of base-form (lemma) strings.
        """
        doc = self.nlp(" ".join(tokens))
        return [
           token.lemma_.lower()
           for token in doc
           if token.is_alpha and token.lemma_ != ""
        ] #is_alpha filters noise → clean tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Filter out any token present in the combined STOPWORDS set.
        KEEP_WORDS are protected — they were subtracted from STOPWORDS earlier,
        so they will never be removed here.
        Note: WORDCLOUD_STOPWORDS is NOT used here — it is only applied inside
        WordCloudAnalysis to filter tokens for word cloud generation.
        """
        return [t for t in tokens if t not in self.stopwords_set]

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN CLEANING PIPELINE (single post)
    # ══════════════════════════════════════════════════════════════════════════

    def clean_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Full cleaning pipeline for ONE post.

        ORDER OF OPERATIONS (important — each step feeds the next):
          1. standardize_text  → lowercase, collapse newlines
          2. remove_emojis     → strip emoji (pass 1)
          3. replace_urls      → swap URLs with ' URL '
          4. replace_mentions  → swap @user with ' PEOPLE '
          5. extract_hashtags  → save #tags, remove from text
          6. Regex cleanup     → keep only \\w, \\s, !, ?, -, '
                                 (flags !, ? must survive for punct counting;
                                  - and ' are technically redundant but harmless)
                                 re.ASCII ensures \\w = [a-zA-Z0-9_] only,
                                 so stray Unicode/emoji chars are wiped
          7. remove_emojis     → pass 2 safety net (catches anything the regex
                                 accidentally let through as a \\w character)
          8. Collapse spaces   → tidy up double spaces

        RETURNS
          (cleaned_text, hashtag_list)
          → stored in df["cleaned_text"] and df["hashtags"]
        """
        text = self.standardize_text(text)
        text = self.remove_emojis(text)          # pass 1
        text = self.replace_urls(text)
        text = self.replace_mentions(text)
        hashtags, text = self.extract_hashtags(text)
        text = re.sub(r"[^\w\s!?\-']", "", text)  
        text = self.remove_emojis(text)          # pass 2 — catches emoji that survived as \w
        text = re.sub(r"\s+", " ", text).strip()
        return text, hashtags

    def preprocess(self, cleaned_text: str) -> List[str]:
        """
        Convenience method: run the full linguistic pipeline in one call.
          tokenize(cleaned_text) → lemmatize(tokens) → remove_stopwords(lemmas)
        Result: a list of clean, meaningful French lemmas ready for NLP analysis.
        This is Branch 2 — runs AFTER punctuation counts are already saved.
        Uses STOPWORDS (not WORDCLOUD_STOPWORDS) — negations and intensity
        modifiers are preserved in the token output.
        """
        tokens = self.tokenize(cleaned_text)
        lemmas = self.lemmatize(tokens)
        return self.remove_stopwords(lemmas)

    # ══════════════════════════════════════════════════════════════════════════
    # APPLY TO ENTIRE DATAFRAME
    # ══════════════════════════════════════════════════════════════════════════

    def fit_transform(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Apply the full cleaning + feature extraction pipeline to every row.

        TWO INDEPENDENT BRANCHES FROM cleaned_text
        ──────────────────────────────────────────
        Branch 1 (Statistical):
          cleaned_text → char_count, text_length, question_count,
                         exclamation_count, ellipsis_count, punct_count
          Punctuation counts are extracted FIRST, before lemmatisation.
          cleaned_text is NEVER overwritten, so counts are safe.

        Branch 2 (Linguistic):
          cleaned_text → preprocess() → tokens → text_nostop
          is_alpha in tokenize() discards all punctuation here,
          so word clouds and n-grams are pure content words only.

        emoji_count and emoticon_count are computed from the ORIGINAL raw text
        (before cleaning) so the signal is not lost by the cleaning steps.
        """
        df = df.copy()

        # ── Step 1: clean text + extract hashtags ─────────────────────────────
        # clean_text returns a (cleaned_text, hashtags) tuple per row.
        # We apply it once and unpack into two columns to avoid running it twice.
        cleaned_results    = df[text_col].apply(self.clean_text)
        df["cleaned_text"] = cleaned_results.apply(lambda x: x[0])  # full cleaned sentence
        df["hashtags"]     = cleaned_results.apply(lambda x: x[1])  # list of #tags

        # ── Diagnostic: URLs, @mentions, hashtags ─────────────────────────────
        # Counted from the ORIGINAL raw text (df[text_col]) before cleaning
        # removes them, so the counts reflect the actual dataset content.
        #
        # a. URL count — both http/https and bare www. addresses
        #    replace_urls() swaps these with ' URL ' token to reduce noise
        #    while preserving sentence structure.
        url_count   = df[text_col].apply(
            lambda x: len(re.findall(r'https?://\S+|www\.\S+', x))
        ).sum()
        posts_w_url = (df[text_col].str.contains(r'https?://|www\.', regex=True)).sum()

        # b. @mention count — confirmed zero in this dataset.
        #    replace_mentions() still runs for robustness but produces no
        #    replacements. The count below formally documents this finding.
        mention_count   = df[text_col].apply(
            lambda x: len(re.findall(r'@\w+', x))
        ).sum()
        posts_w_mention = (df[text_col].str.contains(r'@\w+', regex=True)).sum()

        # c. Hashtag count — from the hashtags column created in Step 1 above.
        #    Hashtags are extracted BEFORE cleaning so they are preserved as a
        #    separate feature for potential downstream analysis.
        hashtag_count   = df["hashtags"].apply(len).sum()
        posts_w_hashtag = (df["hashtags"].apply(len) > 0).sum()

        print(f"\n── Noise Element Counts (from original text) ──")
        print(f"   URLs found      : {url_count}  (in {posts_w_url} posts)")
        print(f"   @mentions found : {mention_count}  (in {posts_w_mention} posts)"
              + ("  ← none in dataset, as expected" if mention_count == 0 else ""))
        print(f"   #hashtags found : {hashtag_count}  (in {posts_w_hashtag} posts)")

        # ── Step 2: tokenise + lemmatise + de-stop (Branch 2) ─────────────────
        # Runs preprocess() on cleaned_text → returns a list of lemmas.
        # Uses STOPWORDS — negations and intensity modifiers are kept here.
        df["tokens"] = df["cleaned_text"].apply(self.preprocess)

        # ── Step 3: surface-level statistics (Branch 1) ───────────────────────
        # All extracted from cleaned_text BEFORE tokens are joined back.
        # char_count counts all characters including spaces and kept punctuation.
        df["char_count"]        = df["cleaned_text"].apply(len)
        # text_length counts words by splitting on whitespace.
        df["text_length"]       = df["cleaned_text"].apply(lambda x: len(x.split()))
        # punct_count is the sum of all three punctuation types below.
        df["punct_count"]       = df["cleaned_text"].apply(
            lambda x: x.count('?') + x.count('!') + x.count('...')
        )
        df["question_count"]    = df["cleaned_text"].apply(lambda x: x.count('?'))
        df["exclamation_count"] = df["cleaned_text"].apply(lambda x: x.count('!'))
        df["ellipsis_count"]    = df["cleaned_text"].apply(lambda x: x.count('...'))

        # ── Step 4: join tokens back into a string for NLP graphs ─────────────
        # e.g. ["sentir", "vide", "douleur"] → "sentir vide douleur"
        # text_nostop is used by: co-occurrence (G5), n-grams (G6)
        # Note: word clouds (G4) use df["tokens"] directly, filtered with
        # WORDCLOUD_STOPWORDS — NOT this text_nostop string.
        df["text_nostop"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

        # ── Step 5: emoji & emoticon counts from the ORIGINAL raw text ─────────
        # We use df[text_col] (not cleaned_text) here so emojis removed during
        # cleaning are still counted — the signal must not be lost.
        df["emoji_count"]    = df[text_col].apply(lambda x: len(emoji.emoji_list(x)))
        df["emoticon_count"] = df[text_col].apply(
            lambda x: sum(
                len(re.findall(p, x, re.IGNORECASE))
                for p in self.emoticon_patterns
            )
        )

        # ── Step 6: emoji removal verification ────────────────────────────────
        # Compare emoji count in ORIGINAL text (emoji_count, Step 5) vs
        # CLEANED text to confirm the pipeline removed all emojis successfully.
        # A temporary column is created and immediately dropped after the check.
        df["emoji_count_after"] = df["cleaned_text"].apply(
            lambda x: len(emoji.emoji_list(x))
        )
        total_before  = df["emoji_count"].sum()         # emojis in original text
        total_after   = df["emoji_count_after"].sum()   # emojis remaining after cleaning
        total_removed = total_before - total_after

        print(f"\n── Emoji Removal Summary ──")
        print(f"   Emojis in original text  : {total_before}")
        print(f"   Emojis after cleaning    : {total_after}")
        print(f"   Emojis removed           : {total_removed}")
        if total_after == 0:
            print(f"   ✓ All emojis successfully removed")
        else:
            print(f"   ⚠️  {total_after} emojis still present — check remove_emojis()")

        # Drop the temporary column — only needed for this verification step
        df.drop(columns=["emoji_count_after"], inplace=True)

        print(f"\n[TextCleaner] Cleaned & tokenized {len(df)} rows")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN EXECUTION — load, deduplicate & clean data
# ══════════════════════════════════════════════════════════════════════════════

# Step 1: Load raw French rows (with duplicates removed inside DataLoader)
df_raw = DataLoader(cfg).load()

# Step 2: Apply full cleaning + feature extraction pipeline
# df is the enriched DataFrame with all analysis-ready columns
cleaner = TextCleaner()
df      = cleaner.fit_transform(df_raw, cfg.TEXT_COL)

# Step 3: Drop any rows where the label is missing
# Missing labels would cause errors in every groupby/plot that uses LABEL_COL
missing_labels = df[cfg.LABEL_COL].isna().sum()
print(f"\n── Missing Values in '{cfg.LABEL_COL}' ──")
print(f"   Missing : {missing_labels}")
print(f"   Total   : {len(df)}")
if missing_labels > 0:
    df = df.dropna(subset=[cfg.LABEL_COL]).reset_index(drop=True)
    print(f"   Dropped rows with missing labels. Remaining: {len(df)}")
else:
    print("   No missing values found ✓")

# Step 4: Diagnostic — confirm emoji/emoticon signal level
# Result: <1% of posts have emojis, 0% have emoticons → dataset is formal/synthetic.
# This justifies the warning added to Graph 10.
print("\n── Emoji & Emoticon Diagnostics ──")
print(df["emoji_count"].value_counts())
print(df["emoticon_count"].value_counts())
print(f"Posts with any emoji:    {(df['emoji_count'] > 0).sum()}")
print(f"Posts with any emoticon: {(df['emoticon_count'] > 0).sum()}")

df.head(3)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Label Distribution
# ══════════════════════════════════════════════════════════════════════════════
class LabelDistribution:
    """
    WHAT IT SHOWS
      How many posts belong to each mental_state label.
      Two views: bar chart (absolute counts) + pie chart (proportions with n=).

    COLUMNS USED
      df["mental_state"] — the target label column

    OUTPUT
      01_label_distribution.png
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        # Count posts per label, sorted descending
        counts = df[self.cfg.LABEL_COL].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Label Distribution — mental_state", fontsize=14, fontweight="bold")

        # ── Left: horizontal bar chart with count annotations ─────────────────
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

        # ── Right: pie chart showing both % and absolute count ────────────────
        def make_autopct(values):
            """Custom autopct: show 'X.X%\n(n=NNN)' inside each pie slice."""
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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Text Length Analysis
# ══════════════════════════════════════════════════════════════════════════════
class TextLengthAnalysis:
    """
    WHAT IT SHOWS
      Distribution of post lengths (word count and character count) per label.
      Three sub-plots:
        2a — histogram of word count per label (with median line)
        2b — boxplot of word count per label (with annotated statistics)
        2c — boxplot of character count per label (with annotated statistics)

    COLUMNS USED
      df["text_length"]  — word count of cleaned_text
      df["char_count"]   — character count of cleaned_text
      df["mental_state"] — used to split data by label

    WHY text_length AND char_count SEPARATELY?
      text_length measures post length in words (linguistic unit).
      char_count measures raw length in characters (includes spaces + punctuation).
      They can diverge: a post with many long words has high char_count
      but moderate text_length.

    OUTPUT
      02a_textlength_histogram_by_label.png
      02b_textlength_boxplot_by_label.png
      02c_charcount_boxplot_by_label.png
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> None:
        labels = df[self.cfg.LABEL_COL].unique()
        colors = sns.color_palette(self.cfg.PALETTE, len(labels))

        # ── Plot 2a: Histogram with median line per label ──────────────────────
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
        fig.suptitle("Text Length Distribution by Label", fontsize=14, fontweight="bold")

        if len(labels) == 1:
            axes = [axes]   # ensure axes is always iterable

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["text_length"]

            counts, bins, patches = ax.hist(
                subset, bins=30, color=color, edgecolor="white", alpha=0.85
            )

            # Red dashed median line helps quickly spot the central tendency
            median_val = subset.median()
            ax.axvline(median_val, color="red", linestyle="--", linewidth=1.8,
                       label=f"Median = {median_val:.0f}")
            ax.legend(fontsize=9)

            # Annotate bar heights (frequency count above each bar)
            for count, patch in zip(counts, patches):
                if count > 0:
                    ax.text(
                        patch.get_x() + patch.get_width() / 2,
                        patch.get_height() + max(counts) * 0.01,
                        f"{int(count)}", ha="center", va="bottom", fontsize=8
                    )

            ax.set_title(f"{label}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Text Length")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        self.helper.save("02a_textlength_histogram_by_label.png")

        # ── Plot 2b: Boxplot of word count per label ───────────────────────────
        # Shows min, Q1, median, mean, Q3, max directly on the plot.
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
        fig.suptitle("Text Length Boxplot by Label", fontsize=14, fontweight="bold")

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["text_length"]

            ax.boxplot(
                subset,
                patch_artist=True,
                boxprops=dict(facecolor=color, color="gray"),
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker='o', markersize=3, markerfacecolor=color, alpha=0.4),
            )

            mn = subset.min();   q1 = subset.quantile(0.25)
            median = subset.median(); mean = subset.mean()
            q3 = subset.quantile(0.75); mx = subset.max()

            # Annotate each statistic on the left or right of the box
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

        # ── Plot 2c: Boxplot of character count per label ──────────────────────
        # Same structure as 2b but uses char_count instead of text_length.
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
        fig.suptitle("Char Count Distribution by Label", fontsize=14, fontweight="bold")

        for ax, label, color in zip(axes, labels, colors):
            subset = df[df[self.cfg.LABEL_COL] == label]["char_count"]

            ax.boxplot(
                subset,
                patch_artist=True,
                boxprops=dict(facecolor=color, color="gray"),
                medianprops=dict(color="black", linewidth=2),
                flierprops=dict(marker='o', markersize=3, markerfacecolor=color, alpha=0.4),
            )

            mn = subset.min();   q1 = subset.quantile(0.25)
            median = subset.median(); mean = subset.mean()
            q3 = subset.quantile(0.75); mx = subset.max()

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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3a — Punctuation Analysis (normalised bar chart)
# ══════════════════════════════════════════════════════════════════════════════
class PunctuationAnalysis:
    """
    WHAT IT SHOWS
      Average usage of ?, !, and ... per sentence, grouped by label.
      Normalising by sentence count prevents longer posts from dominating.

    WHY NORMALISE BY SENTENCES?
      A post with 10 sentences and 5 question marks uses fewer questions
      per sentence than a post with 2 sentences and 3 question marks.
      Raw counts would be misleading — normalised rates are comparable.

    COLUMNS USED
      df["cleaned_text"]        — for sentence tokenisation via sent_tokenize
      df["question_count"]      — pre-computed ? count
      df["exclamation_count"]   — pre-computed ! count
      df["ellipsis_count"]      — pre-computed ... count
      df["mental_state"]        — to group by label

    OUTPUT
      03_punctuation_normalized.png
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        punct_cols = ['question_count', 'exclamation_count', 'ellipsis_count']

        # Count sentences per post (minimum 1 to avoid division by zero)
        sentence_counts = df['cleaned_text'].apply(
            lambda t: max(len(sent_tokenize(t)), 1)
        )

        # Build a summary table: for each label, compute avg count / avg sentences
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

        # Grouped bar chart: one group per punctuation type, one bar per label
        x      = np.arange(len(norm_df.columns))
        labels = norm_df.index.tolist()
        n      = len(labels)
        width  = 0.35
        colors = sns.color_palette(self.cfg.PALETTE, n)

        for i, (label, color) in enumerate(zip(labels, colors)):
            offset = (i - n / 2 + 0.5) * width
            bars   = ax.bar(x + offset, norm_df.loc[label],
                            width=width, label=label, color=color, edgecolor="white")
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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3b — Punctuation Table
# ══════════════════════════════════════════════════════════════════════════════
class PunctuationTable:
    """
    WHAT IT SHOWS
      Same normalised punctuation values as Graph 3a but as a formatted table.
      Useful for exact numeric comparison and for including in a written report.

    COLUMNS USED
      Same as PunctuationAnalysis (Graph 3a).

    OUTPUT
      03b_punctuation_table.png
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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 4 — Word Clouds per Label
# ══════════════════════════════════════════════════════════════════════════════
class WordCloudAnalysis:
    """
    WHAT IT SHOWS
      One word cloud per label — a visual summary of the most frequent
      content words used by posts in that mental-health category.
      Word size is proportional to frequency.

    COLUMNS USED
      df["tokens"]       — lemmatised token lists filtered with WORDCLOUD_STOPWORDS
      df["mental_state"] — to split posts by label

    WHY tokens AND NOT text_nostop?
      text_nostop was built using STOPWORDS (which keeps negations and intensity
      modifiers). For word clouds we apply WORDCLOUD_STOPWORDS which additionally
      removes those high-frequency words. We therefore filter df["tokens"]
      directly rather than using the pre-built text_nostop string.

    WHY WORDCLOUD_STOPWORDS AND NOT STOPWORDS?
      Negations (pas, jamais) and intensity modifiers (trop, tellement) kept
      in STOPWORDS dominate word clouds visually and hide the actual content
      words (douleur, espoir, anxiété …). WORDCLOUD_STOPWORDS removes them
      for a more informative visual, without affecting any other analysis.

    OUTPUT
      04_wordclouds_per_label.png
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
            # Filter each post's token list using WORDCLOUD_STOPWORDS to remove
            # negations and intensity modifiers that dominate the visual.
            # Then join all filtered tokens into one large string for the cloud.
            tokens_filtered = (
                df[df[self.cfg.LABEL_COL] == label]["tokens"]
                .apply(lambda t: [w for w in t if w not in WORDCLOUD_STOPWORDS])
            )
            text = " ".join([" ".join(t) for t in tokens_filtered])

            if not text.strip():
                axes[i].axis("off")
                continue

            wc = WordCloud(
                width=600, height=350,
                background_color="white",
                colormap=cmaps[i % len(cmaps)],
                max_words=100,
                collocations=False,   # avoids the same bigram appearing repeatedly
            ).generate(text)

            axes[i].imshow(wc, interpolation="bilinear")
            axes[i].axis("off")
            axes[i].set_title(str(label), fontsize=12, fontweight="bold")

        # Hide any leftover empty subplot cells
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        path = self.helper.save("04_wordclouds_per_label.png")
        return [path]


WordCloudAnalysis(cfg, helper).analyse(df)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 5 — Co-occurrence Analysis
# ══════════════════════════════════════════════════════════════════════════════
class CoOccurrenceAnalysis:
    """
    WHAT IT SHOWS
      Which pairs of words appear together most often within the same post,
      per label. High co-occurrence suggests words are semantically linked
      in that mental-health context.

    HOW CO-OCCURRENCE IS COMPUTED
      For each post (text_nostop), take the SET of unique words (so duplicates
      within one post don't inflate counts), then count every sorted pair
      across all posts using combinations().

    COLUMNS USED
      df["text_nostop"]  — stop-word-free tokens string
      df["mental_state"] — to split by label

    OUTPUT
      05_cooccurrence_<label>.png  (one file per label)
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def _cooccurrence(self, texts, top_n):
        """Count all word pairs across all posts and return the top_n most common."""
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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 6 — Common Words / Bigrams / Trigrams
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.feature_extraction.text import CountVectorizer


class CommonWordsAnalysis:
    """
    WHAT IT SHOWS
      Top N most frequent unigrams (words), bigrams (2-word phrases), and
      trigrams (3-word phrases) per label.

    WHY CountVectorizer INSTEAD OF Counter?
      CountVectorizer handles n-gram extraction natively, applies min_df
      (ignore terms appearing in <2 posts) and max_df (ignore terms in >95%
      of posts) to filter noise and trivially common terms automatically.

    COLUMNS USED
      df["text_nostop"]  — stop-word-free lemmatised tokens
      df["mental_state"] — to split by label

    OUTPUT
      06_common_words_per_label.png   (unigrams)
      06_bigrams_per_label.png        (bigrams)
      06_trigrams_per_label.png       (trigrams)
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def _get_ngram_freq(self, texts, n):
        """
        Fit a CountVectorizer for n-grams of size n on the given texts.
        Returns a DataFrame of (ngram, count) sorted by frequency, top N only.
        """
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            min_df=2,     # ignore n-grams that appear in fewer than 2 posts (noise)
            max_df=0.95   # ignore n-grams in >95% of posts (too generic)
        )
        X     = vectorizer.fit_transform(texts)
        freqs = X.toarray().sum(axis=0)
        df    = pd.DataFrame({
            "ngram": vectorizer.get_feature_names_out(),
            "count": freqs
        })
        return df.sort_values("count", ascending=False).head(self.cfg.TOP_N_WORDS)

    def _plot_ngrams(self, df: pd.DataFrame, n: int, title_prefix: str, filename: str) -> str:
        """
        Create a grid of horizontal bar charts — one subplot per label.
        Each subplot shows the top N n-grams for that label.
        """
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

            sns.barplot(data=top_ngrams, x="count", y="ngram",
                        palette="rocket", ax=axes[i])

            axes[i].set_title(str(label), fontsize=11, fontweight="bold")
            axes[i].set_xlabel("Frequency")
            axes[i].set_ylabel("")

            for bar, val in zip(axes[i].patches, top_ngrams["count"]):
                axes[i].text(
                    bar.get_width() + max(top_ngrams["count"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8,
                )

            axes[i].set_xlim(0, max(top_ngrams["count"]) * 1.12)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        return self.helper.save(filename)

    def analyse(self, df: pd.DataFrame) -> list:
        """Run for unigrams (n=1), bigrams (n=2), and trigrams (n=3)."""
        return [
            self._plot_ngrams(df, 1, "Common Words", "06_common_words_per_label.png"),
            self._plot_ngrams(df, 2, "Bigrams",       "06_bigrams_per_label.png"),
            self._plot_ngrams(df, 3, "Trigrams",      "06_trigrams_per_label.png"),
        ]


CommonWordsAnalysis(cfg, helper).analyse(df)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 7 — Category Distribution
# ══════════════════════════════════════════════════════════════════════════════
class CategoryDistribution:
    """
    WHAT IT SHOWS
      How many posts fall into each mental-health CATEGORY (e.g. Depression,
      Anxiety, Bipolar …). This is different from LABEL (Healthy/Unhealthy).

    COLUMNS USED
      df["category"] — the mental-health category column

    OUTPUT
      07_category_distribution.png
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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 8 — Category × Label Heatmap
# ══════════════════════════════════════════════════════════════════════════════
class CategoryLabelHeatmap:
    """
    WHAT IT SHOWS
      A cross-tabulation (contingency table) of category vs. mental_state,
      visualised as a colour-coded heatmap.
      Reveals which categories contribute more Healthy vs. Unhealthy posts.
      Row and column totals are included for full context.

    COLUMNS USED
      df["category"]     — mental-health category
      df["mental_state"] — binary label (Healthy / Unhealthy)

    OUTPUT
      08_category_label_heatmap.png
    """

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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 8b — Per-Category Healthy vs Unhealthy Word Comparison (printed)
# ══════════════════════════════════════════════════════════════════════════════
# This block prints (not plots) the top 15 words for Healthy and Unhealthy
# posts within each category. Used for qualitative inspection of vocabulary
# differences — e.g. does "Depression / Healthy" use different words than
# "Depression / Unhealthy"?
#
# COLUMNS USED
#   df["category"]     — to loop over each mental-health category
#   df["mental_state"] — to split into Healthy / Unhealthy
#   df["text_nostop"]  — the clean word tokens for frequency counting
for category in df["category"].unique():
    subset = df[df["category"] == category]

    healthy_words   = " ".join(subset[subset["mental_state"] == "Healthy"]["text_nostop"])
    unhealthy_words = " ".join(subset[subset["mental_state"] == "Unhealthy"]["text_nostop"])

    healthy_freq   = Counter(healthy_words.split()).most_common(15)
    unhealthy_freq = Counter(unhealthy_words.split()).most_common(15)

    print(f"\n=== {category} ===")
    print("Healthy top words:",   healthy_freq)
    print("Unhealthy top words:", unhealthy_freq)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 10 — Emoji & Emoticon Analysis
# ══════════════════════════════════════════════════════════════════════════════
class EmojiEmoticonAnalysis:
    """
    WHAT IT SHOWS
      Distribution and average usage of emojis and emoticons per label.
      Includes a pie chart showing the proportion of posts that contain emojis.

    IMPORTANT CAVEAT
      <1% of posts contain emojis and 0% contain emoticons, so the results
      are not statistically meaningful. The warning is shown in the title.
      The analysis is retained to document this finding transparently.

    COLUMNS USED
      df["emoji_count"]    — emoji count from ORIGINAL raw text
      df["emoticon_count"] — ASCII emoticon count from ORIGINAL raw text
      df["mental_state"]   — to group averages by label

    OUTPUT
      10_emoji_emoticon.png
      Layout: 2 rows × 3 cols
        [0,0] distribution histogram — emojis
        [0,1] distribution histogram — emoticons
        [0,2] pie chart — proportion of posts with/without emoji
        [1,0] avg emoji count by label (bar chart)
        [1,1] avg emoticon count by label (bar chart)
        [1,2] empty (unused)
    """

    def __init__(self, cfg: Config, helper: PlotHelper):
        self.cfg    = cfg
        self.helper = helper

    def analyse(self, df: pd.DataFrame) -> str:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Emoji & Emoticon Analysis\n"
            "⚠️ <1% of posts contain emojis — results not statistically meaningful",
            fontsize=13, fontweight="bold"
        )

        # ── Row 0, cols 0–1: histograms with mean + median lines ──────────────
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
                    ax.text(patch.get_x() + patch.get_width() / 2,
                            h + 0.5, str(int(h)),
                            ha="center", va="bottom", fontsize=8)

        # ── Row 0, col 2: pie chart — emoji presence ──────────────────────────
        ax_pie         = axes[0, 2]
        emoji_presence = (df["emoji_count"] > 0).value_counts()
        no_emoji       = emoji_presence.get(False, 0)
        has_emoji      = emoji_presence.get(True,  0)
        sizes          = [no_emoji, has_emoji]
        pie_labels     = [f"No Emoji\n(n={no_emoji})", f"Contains Emoji\n(n={has_emoji})"]

        ax_pie.pie(sizes, labels=pie_labels, autopct="%1.2f%%",
                   colors=["#d3d3d3", "#F4A460"], startangle=90)
        ax_pie.set_title("Emoji Presence Distribution")

        # ── Row 1, cols 0–1: average per label ───────────────────────────────
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
                ax.text(bar.get_width() + avg.values.max() * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9)
            ax.set_xlim(0, avg.values.max() * 1.15)

        # ── Row 1, col 2: unused cell ─────────────────────────────────────────
        axes[1, 2].axis("off")

        plt.tight_layout()
        return self.helper.save("10_emoji_emoticon.png")


EmojiEmoticonAnalysis(cfg, helper).analyse(df)


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY — save cleaned CSV + emoji log
# ══════════════════════════════════════════════════════════════════════════════

# Save the cleaned, feature-enriched DataFrame so downstream modelling steps
# (e.g. TF-IDF, BERT fine-tuning) can load it without re-running this pipeline.
df.to_csv(f"{cfg.OUTPUT_DIR}/french_cleaned.csv", index=False, encoding="utf-8-sig")

# ── Count emojis in the ORIGINAL (pre-cleaning) dataset ───────────────────────
# This uses df_raw (not df) so cleaning doesn't affect the emoji inventory.
# Saved both to console and to a text file for the written report.
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

# ── Append hashtags found in the dataset to the same log file ─────────────────
# Flattens all hashtag lists from df["hashtags"] into a single Counter.
# Appended to emojis_hashtags_found.txt so both noise elements are in one reference file.
all_hashtags   = Counter([tag for tags in df["hashtags"] for tag in tags])

with open(emoji_txt_path, "a", encoding="utf-8") as f:
    f.write(f"\n\nHashtags found in dataset ({len(all_hashtags)} unique)\n")
    f.write("=" * 45 + "\n")
    for tag, count in all_hashtags.most_common():
        f.write(f"  {tag}  →  {count} times\n")

print(f"\n── Hashtags found in dataset ({len(all_hashtags)} unique) ──")
for tag, count in all_hashtags.most_common():
    print(f"   {tag}  →  {count} times")

print(f"\n── Emojis found in ORIGINAL dataset ({len(emoji_counts)} unique) ──")
for em, count in emoji_counts.most_common():
    print(f"   {em}  →  {count} times")
print(f"\n📄 Emoji list saved to: {emoji_txt_path}")

# ── Final summary printout ────────────────────────────────────────────────────
print("=" * 55)
print("  FRENCH EDA PIPELINE — COMPLETE")
print("=" * 55)

saved = [f for f in os.listdir(cfg.OUTPUT_DIR) if f.endswith(".png")]
print(f"\n📊 {len(saved)} plots saved to '{cfg.OUTPUT_DIR}/':")
for f in sorted(saved):
    print(f"   • {f}")

print(f"\n📄 Cleaned CSV  : {cfg.OUTPUT_DIR}/french_cleaned.csv")
print(f"📄 Emoji list   : {emoji_txt_path}")

# ── Lemmatisation example from the actual dataset ─────────────────────────────
# Pick 3 real posts from the dataset and show the full lemmatisation pipeline:
# original text → tokens → lemmas → after stopword removal

print("── Lemmatisation Examples from Dataset ──\n")

for i in range(3):
    raw        = df_raw[cfg.TEXT_COL].iloc[i]        # original raw post
    cleaned    = df["cleaned_text"].iloc[i]           # after cleaning
    tokens_raw = cleaner.tokenize(cleaned)            # spaCy tokens (no lemma yet)
    lemmas     = cleaner.lemmatize(tokens_raw)        # lemmatised
    final      = df["tokens"].iloc[i]                 # after stopword removal

    print(f"POST {i+1}")
    print(f"  Original  : {raw[:120]}")
    print(f"  Cleaned   : {cleaned[:120]}")
    print(f"  Tokens    : {tokens_raw[:10]}")
    print(f"  Lemmas    : {lemmas[:10]}")
    print(f"  Final     : {final[:10]}")
    print()


print("\n✅ All done!")

