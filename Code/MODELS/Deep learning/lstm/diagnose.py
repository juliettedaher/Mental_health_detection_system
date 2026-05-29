"""
diagnose.py
-----------
Run this in your lstm/ folder to find the real problem.
It checks your actual data, tokenizer output, and gradient flow.

Usage:
    python diagnose.py --config config.yaml
"""

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

from dataset   import load_french_dataset
from tokenizer import Tokenizer
from lstm      import get_model


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\n" + "="*60)
    print("  DIAGNOSIS REPORT")
    print("="*60)

    # ── 1. Load data ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, le = load_french_dataset(
        path=cfg["paths"]["data"],
        text_col=cfg["dataset"]["text_col"],
        label_col=cfg["dataset"]["label_col"],
        test_size=cfg["dataset"]["test_size"],
        random_state=cfg["dataset"]["random_state"],
    )

    # ── 2. Check raw text samples ────────────────────────────────
    print("\n[1] SAMPLE RAW TEXTS (first 5 from each class):")
    X_arr = np.array(list(X_train))
    y_arr = np.array(y_train)

    for cls_idx, cls_name in enumerate(le.classes_):
        mask    = y_arr == cls_idx
        samples = X_arr[mask][:5]
        print(f"\n  --- {cls_name} ---")
        for s in samples:
            print(f"    '{s[:120]}'")

    # ── 3. Text length distribution ──────────────────────────────
    print("\n[2] TEXT LENGTH DISTRIBUTION (words):")
    lengths = [len(str(t).split()) for t in X_train]
    print(f"  min={min(lengths)}  max={max(lengths)}  "
          f"mean={np.mean(lengths):.1f}  median={np.median(lengths):.1f}")
    print(f"  texts with <=5 words  : {sum(l<=5 for l in lengths)} "
          f"({100*sum(l<=5 for l in lengths)/len(lengths):.1f}%)")
    print(f"  texts with <=10 words : {sum(l<=10 for l in lengths)} "
          f"({100*sum(l<=10 for l in lengths)/len(lengths):.1f}%)")

    # ── 4. Tokenizer output ──────────────────────────────────────
    print("\n[3] TOKENIZER CHECK:")
    tok = Tokenizer(
        max_vocab=cfg.get("tokenizer", {}).get("max_vocab", 20000),
        max_len=cfg.get("tokenizer", {}).get("max_len", 100),
    )
    tok.fit(X_train)

    encoded = np.array([tok.encode(t) for t in X_train])
    unk_rate = (encoded == 1).sum() / encoded.size
    pad_rate = (encoded == 0).sum() / encoded.size

    print(f"  Vocab size : {len(tok.word2idx):,}")
    print(f"  UNK rate   : {unk_rate:.2%}  ← should be <5%")
    print(f"  PAD rate   : {pad_rate:.2%}  ← high if texts are short")

    # show a sample encoding
    sample_text = str(list(X_train)[0])
    sample_enc  = tok.encode(sample_text)
    print(f"\n  Sample text  : '{sample_text[:80]}'")
    print(f"  Sample tokens: {sample_enc[:15]} ...")

    # ── 5. Class balance after encoding ─────────────────────────
    print("\n[4] CLASS BALANCE:")
    c = Counter(y_train)
    total = len(y_train)
    for k, v in sorted(c.items()):
        print(f"  Class {k} ({le.classes_[k]}): {v} ({100*v/total:.1f}%)")

    # ── 6. Gradient flow test ────────────────────────────────────
    print("\n[5] GRADIENT FLOW (1 batch, untrained model):")
    cfg["model"]["vocab_size"] = len(tok.word2idx)
    model  = get_model(cfg)
    device = torch.device("cpu")

    X_batch = torch.tensor(encoded[:32], dtype=torch.long)
    y_batch = torch.tensor(np.array(y_train[:32]), dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    logits    = model(X_batch)
    loss      = criterion(logits, y_batch)
    loss.backward()

    print(f"  Loss after 1 forward pass : {loss.item():.4f}")
    print(f"  Logits sample (first 5)   : {logits[:5].detach().numpy().round(4)}")
    print(f"  Predicted classes         : {logits.argmax(1)[:10].numpy()}")

    total_grad = 0.0
    zero_grads = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.abs().mean().item()
            total_grad += g
            if g < 1e-8:
                zero_grads += 1
                print(f"  ⚠ DEAD GRADIENT: {name}  (mean abs grad = {g:.2e})")

    print(f"\n  Total mean abs grad : {total_grad:.6f}")
    print(f"  Dead gradient layers: {zero_grads}")

    if zero_grads > 0:
        print("  → Some layers have essentially zero gradient. Model won't learn.")
    else:
        print("  → Gradients look healthy.")

    # ── 7. Quick sanity: can the model overfit 32 samples? ──────
    print("\n[6] OVERFIT SANITY CHECK (32 samples, 100 steps):")
    model     = get_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_batch).argmax(1)
    acc = (preds == y_batch).float().mean().item()
    print(f"  Accuracy on 32 training samples after 100 steps: {acc:.2%}")
    if acc > 0.85:
        print("  → Model CAN learn. Problem is in training config or data pipeline.")
    else:
        print("  → Model CANNOT overfit 32 samples. Architecture or data problem.")

    print("\n" + "="*60)
    print("  END OF DIAGNOSIS")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default=r"C:\Users\Admin\Documents\FYP\french dataset\Code\MODELS\lstm\config.yaml"
    )
    args = parser.parse_args()
    main(args.config) 