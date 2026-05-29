"""
tfidf_vectorizer.py
-------------------
Wraps the TF-IDF logic from the SVM notebook into a clean class.

Key rule: fit ONLY on training data, then transform train and test separately.
This prevents data leakage (test data must never influence the vocabulary).

Usage
-----
    from tfidf_vectorizer import TfidfVectorizerWrapper

    vec = TfidfVectorizerWrapper()
    X_train_tfidf = vec.fit_transform(X_train_texts)  # fits + transforms
    X_test_tfidf  = vec.transform(X_test_texts)        # transforms only

    # Save / load the fitted vectorizer
    vec.save("vectorizers/tfidf_vectorizer.pkl")
    vec2 = TfidfVectorizerWrapper.load("vectorizers/tfidf_vectorizer.pkl")

    # Single-sentence inference (for the website / predictor.py)
    features = vec.transform_one("je me sens très bien aujourd'hui")
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse


class TfidfVectorizerWrapper:
    """
    TF-IDF vectorizer with fit/transform separation and save/load helpers.

    Parameters
    ----------
    max_features : vocabulary size cap         (default: 5000)
    ngram_range  : min and max n-gram length   (default: (1, 2) — unigrams + bigrams)

    Attributes
    ----------
    vectorizer   : fitted sklearn TfidfVectorizer (available after fit_transform)
    """

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, texts) -> np.ndarray:
        """
        Fit the vectorizer on training texts AND transform them.
        Call this ONLY on X_train — never on X_test.

        Parameters
        ----------
        texts : list or pd.Series of strings

        Returns
        -------
        np.ndarray of shape (N, max_features)  — dense matrix
        """
        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        X = self.vectorizer.fit_transform(texts)
        self._is_fitted = True

        X_dense = X.toarray() if issparse(X) else X
        print(f"[TfidfVectorizerWrapper] fit_transform → shape: {X_dense.shape}")
        return X_dense

    # ------------------------------------------------------------------
    def transform(self, texts) -> np.ndarray:
        """
        Transform texts using the already-fitted vocabulary.
        Call this on X_test (and at inference time).

        Parameters
        ----------
        texts : list or pd.Series of strings

        Returns
        -------
        np.ndarray of shape (N, max_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() on training data before transform().")

        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        X = self.vectorizer.transform(texts)
        X_dense = X.toarray() if issparse(X) else X
        print(f"[TfidfVectorizerWrapper] transform → shape: {X_dense.shape}")
        return X_dense

    # ------------------------------------------------------------------
    def transform_one(self, text: str) -> np.ndarray:
        """
        Transform a single sentence — used at inference time (website).

        Parameters
        ----------
        text : raw French string

        Returns
        -------
        np.ndarray of shape (1, max_features)
        """
        return self.transform([text])

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Pickle the fitted vectorizer to disk.

        Parameters
        ----------
        path : file path, e.g. 'vectorizers/tfidf_vectorizer.pkl'
        """
        if not self._is_fitted:
            raise RuntimeError("Fit the vectorizer before saving.")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[TfidfVectorizerWrapper] Saved → {path}")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "TfidfVectorizerWrapper":
        """
        Load a previously saved TfidfVectorizerWrapper from disk.

        Parameters
        ----------
        path : file path, e.g. 'vectorizers/tfidf_vectorizer.pkl'

        Returns
        -------
        TfidfVectorizerWrapper (already fitted)
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[TfidfVectorizerWrapper] Loaded ← {path}")
        return obj
