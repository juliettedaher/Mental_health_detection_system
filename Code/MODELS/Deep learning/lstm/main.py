import argparse
import yaml
import torch
import numpy as np

from dataset   import load_french_dataset
from lstm      import get_model
from trainer   import Trainer
from tokenizer import Tokenizer


def main(config_path: str):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print("  French Mental Health Classifier (LSTM)")
    print(f"  Model     : {cfg['model']['architecture']}")
    print(f"  Embedding : Trainable Embedding Layer")
    print("=" * 60 + "\n")

    # ── Dataset ──────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, label_encoder = load_french_dataset(
        path=cfg["paths"]["data"],
        text_col=cfg["dataset"]["text_col"],
        label_col=cfg["dataset"]["label_col"],
        test_size=cfg["dataset"]["test_size"],
        random_state=cfg["dataset"]["random_state"],
    )

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device: {device}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tok_cfg   = cfg.get("tokenizer", {})
    tokenizer = Tokenizer(
        max_vocab=tok_cfg.get("max_vocab", cfg["model"]["vocab_size"]),
        max_len=tok_cfg.get("max_len", 100),
    )
    tokenizer.fit(X_train)                         # fit on train ONLY

    X_train = np.array([tokenizer.encode(t) for t in X_train])
    X_test  = np.array([tokenizer.encode(t) for t in X_test])

    cfg["model"]["vocab_size"] = len(tokenizer.word2idx)   # update for embedding layer

    # ── Model ─────────────────────────────────────────────────────────────
    model        = get_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Trainable parameters: {total_params:,}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device)
    trainer.fit(X_train, y_train, X_test, y_test, label_encoder=label_encoder)

    print("\n[main] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=r"C:\Users\Admin\Documents\FYP\french dataset\Code\MODELS\Deep learning\lstm\config.yaml",
    )
    args = parser.parse_args()
    main(args.config)