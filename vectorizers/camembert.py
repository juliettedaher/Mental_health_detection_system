"""
camembert_vectorizer.py
-----------------------
Wraps the CamemBERT embedding logic from the SVM notebook into a clean class.

The model is used as a FROZEN feature extractor (no fine-tuning).
Each sentence → 768-dim CLS-token vector.

Usage
-----
    from camembert_vectorizer import CamembertVectorizer

    vec = CamembertVectorizer()                 # loads camembert-base
    X_train_emb = vec.encode_dataset(X_train_texts)   # np.ndarray (N, 768)
    X_test_emb  = vec.encode_dataset(X_test_texts)

    # Save / load embeddings so you don't re-run the 8-min encoding every time
    vec.save_embeddings(X_train_emb, "vectorizers/X_train_cam.npy")
    X_train_emb = vec.load_embeddings("vectorizers/X_train_cam.npy")

    # Single-sentence inference (for the website / predictor.py)
    embedding = vec.encode("je me sens très bien aujourd'hui")  # (768,)
"""

import numpy as np
import torch
from transformers import CamembertTokenizer, CamembertModel
from tqdm import tqdm


class CamembertVectorizer:
    """
    CamemBERT feature extractor — French RoBERTa-based model.

    Attributes
    ----------
    model_name  : HuggingFace model id          (default: 'camembert-base')
    batch_size  : sentences processed per batch  (default: 16)
    max_length  : max tokens per sentence        (default: 128)
    device      : torch.device (auto-detected)
    """

    def __init__(
        self,
        model_name: str = "camembert-base",
        batch_size: int = 16,
        max_length: int = 128,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Auto-detect GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CamembertVectorizer] Using device: {self.device}")

        # Load tokenizer and model from HuggingFace
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model = CamembertModel.from_pretrained(model_name)
        self.model.to(self.device)

        # Freeze weights — we are NOT fine-tuning CamemBERT
        self.model.eval()

    # ------------------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single sentence → 768-dim numpy vector.
        Used at inference time (website / predictor.py).

        Parameters
        ----------
        text : raw French string (already preprocessed / stopwords removed)

        Returns
        -------
        np.ndarray of shape (768,)
        """
        return self.encode_dataset([text])[0]

    # ------------------------------------------------------------------
    def encode_dataset(self, texts, verbose: bool = True) -> np.ndarray:
        """
        Encode a list of sentences → 2-D embedding matrix.

        Parameters
        ----------
        texts   : list or pd.Series of French strings
        verbose : show tqdm progress bar (default: True)

        Returns
        -------
        np.ndarray of shape (len(texts), 768)
        """
        if hasattr(texts, "tolist"):
            texts = texts.tolist()

        all_embeddings = []

        iterator = range(0, len(texts), self.batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="CamemBERT embedding")

        for i in iterator:
            batch = texts[i : i + self.batch_size]

            # Tokenize: truncate long texts, pad short ones, return PyTorch tensors
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # CLS token = first token of last hidden state → sentence summary
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    @staticmethod
    def save_embeddings(embeddings: np.ndarray, path: str) -> None:
        """
        Save embeddings to a .npy file so you don't re-run the 8-min
        encoding every time you train a new model.

        Parameters
        ----------
        embeddings : np.ndarray of shape (N, 768)
        path       : file path, e.g. 'vectorizers/X_train_cam.npy'
        """
        np.save(path, embeddings)
        print(f"[CamembertVectorizer] Embeddings saved → {path}")

    # ------------------------------------------------------------------
    @staticmethod
    def load_embeddings(path: str) -> np.ndarray:
        """
        Load previously saved embeddings from a .npy file.

        Parameters
        ----------
        path : file path, e.g. 'vectorizers/X_train_cam.npy'

        Returns
        -------
        np.ndarray of shape (N, 768)
        """
        emb = np.load(path)
        print(f"[CamembertVectorizer] Embeddings loaded ← {path}  shape: {emb.shape}")
        return emb
