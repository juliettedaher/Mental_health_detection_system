"""
predictor.py
------------
Inference script for LSTM mental health classifier (clean version).

Assumes:
- Model uses nn.Embedding
- Text is tokenized into integer sequences
- Vocabulary is saved in config or dataset preprocessing
"""

import argparse
import os
import yaml
import torch
import numpy as np

from lstm import get_model
from dataset import text_to_sequence   # 🔥 YOU MUST IMPLEMENT THIS


class Predictor:

    def __init__(self, config_path: str = "config.yaml"):

        # ── Load config ───────────────────────────────────────────────
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Load model ───────────────────────────────────────────────
        self.model = get_model(self.cfg).to(self.device)

        ckpt_path = self.cfg["paths"]["checkpoint"]

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}"
            )

        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )

        self.model.eval()

        print(f"[Predictor] Loaded model from {ckpt_path}")

        # ── Class names ───────────────────────────────────────────────
        self.classes = self.cfg["model"]["classes"]

    # ───────────────────────────────────────────────────────────────
    def predict(self, text: str):

        # 🔥 Convert raw text → token sequence
        seq = text_to_sequence(text, self.cfg)

        X = torch.tensor(seq, dtype=torch.long)\
            .unsqueeze(0)\
            .to(self.device)

        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))

        return self.classes[pred_idx], float(probs[pred_idx])


# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="French sentence to classify"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml"
    )

    args = parser.parse_args()

    predictor = Predictor(args.config)
    label, conf = predictor.predict(args.text)

    print("\nText       :", args.text)
    print("Prediction :", label)
    print(f"Confidence : {conf:.2%}")