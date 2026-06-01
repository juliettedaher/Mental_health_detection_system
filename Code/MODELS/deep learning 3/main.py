import argparse
import os
import json
import csv
from datetime import datetime

import yaml
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from lstm    import get_model
from trainer import Trainer
from metrics import evaluate, plot_confusion_matrix

import sys

class Logger:
    """Tees stdout to both the terminal and a log file."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal



def save_results(cfg, fold_results, all_y_true, all_y_pred, le, run_name, results_dir):
    """Save hyperparams + results to JSON and CSV."""

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    accs = [r["accuracy"] for r in fold_results]
    f1s  = [r["f1_macro"] for r in fold_results]

    # ── Build record ──────────────────────────────────────────────
    record = {
        "timestamp":    timestamp,
        "run_name":     run_name,
        "vectorizer":   cfg["vectorizer"],
        "architecture": cfg["model"]["architecture"],
        "input_size":   cfg["model"]["input_size"],
        "hidden_size":  cfg["model"]["hidden_size"],
        "num_layers":   cfg["model"]["num_layers"],
        "dropout":      cfg["model"]["dropout"],
        "n_folds":      cfg.get("cv", {}).get("n_folds", 5),
        "epochs":       cfg["training"]["epochs"],
        "batch_size":   cfg["training"]["batch_size"],
        "learning_rate":cfg["training"]["learning_rate"],
        "weight_decay": cfg["training"]["weight_decay"],
        "clip_grad_norm":cfg["training"]["clip_grad_norm"],
        "patience":     cfg["training"]["patience"],
        "mean_accuracy":round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)),  4),
        "mean_f1_macro":round(float(np.mean(f1s)),  4),
        "std_f1_macro": round(float(np.std(f1s)),   4),
        "overall_accuracy": round(float(accuracy_score(all_y_true, all_y_pred)), 4),
        "overall_f1_macro": round(float(f1_score(all_y_true, all_y_pred,
                                                  average="macro", zero_division=0)), 4),
        "overall_f1_weighted": round(float(f1_score(all_y_true, all_y_pred,
                                                     average="weighted", zero_division=0)), 4),
        "per_fold": fold_results,
    }

    # ── 1. Save JSON (full detail) ────────────────────────────────
    json_path = os.path.join(results_dir, f"{run_name}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[results] JSON saved  → {json_path}")

    # ── 2. Append to CSV (one row per run, easy to compare) ───────
    csv_path = os.path.join(results_dir, "all_results.csv")
    flat = {k: v for k, v in record.items() if k != "per_fold"}
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)
    print(f"[results] CSV updated → {csv_path}")


def main(config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vec_choice = cfg["vectorizer"].lower()
    arch       = cfg["model"]["architecture"].upper()
    n_folds    = cfg.get("cv", {}).get("n_folds", 5)
    run_name   = f"{arch}_{vec_choice.upper()}"

    print("\n" + "=" * 60)
    print("  French Mental Health Classifier — Cross Validation")
    print(f"  Model      : {arch}")
    print(f"  Vectorizer : {vec_choice.upper()}")
    print(f"  Input size : {cfg['model']['input_size']}")
    print(f"  Folds      : {n_folds}")
    print("=" * 60 + "\n")

    # ── 1. Load full dataset ──────────────────────────────────────
    print(f"[main] Loading: {cfg['paths']['data']}")
    df = pd.read_csv(cfg["paths"]["data"])
    df = df.dropna(subset=[cfg["dataset"]["text_col"],
                            cfg["dataset"]["label_col"]])

    le = LabelEncoder()
    y  = le.fit_transform(df[cfg["dataset"]["label_col"]].values)
    X  = df[cfg["dataset"]["text_col"]].values

    print(f"[main] Samples : {len(X)}")
    print(f"[main] Classes : {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

    # ── 2. Pre-vectorize if CamemBERT ────────────────────────────
    if vec_choice == "camembert":
        from camembert import CamembertVectorizer
        full_path = os.path.join(
            os.path.dirname(cfg["paths"]["cam_train_emb"]), "X_full_cam.npy"
        )
        if os.path.exists(full_path):
            print("[main] Loading saved CamemBERT embeddings...")
            vec   = CamembertVectorizer()
            X_raw = vec.load_embeddings(full_path)
        else:
            print("[main] Encoding full dataset with CamemBERT (~8 min)...")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            vec   = CamembertVectorizer()
            X_raw = vec.encode_dataset(X.tolist())
            vec.save_embeddings(X_raw, full_path)
    else:
        X_raw = X   # TF-IDF vectorized inside each fold

    # ── 3. K-Fold CV ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device: {device}\n")

    skf          = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                    random_state=cfg["dataset"]["random_state"])
    fold_results = []
    all_y_true   = []
    all_y_pred   = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), start=1):

        print(f"\n{'─'*55}")
        print(f"  Fold {fold}/{n_folds}")
        print(f"{'─'*55}")

        if vec_choice == "tfidf":
            from tfidf import TfidfVectorizerWrapper
            vec         = TfidfVectorizerWrapper(max_features=cfg["model"]["input_size"])
            X_train_vec = vec.fit_transform(X_raw[train_idx])
            X_val_vec   = vec.transform(X_raw[val_idx])
        else:
            X_train_vec = X_raw[train_idx]
            X_val_vec   = X_raw[val_idx]

        y_train = y[train_idx]
        y_val   = y[val_idx]

        model   = get_model(cfg).to(device)
        trainer = Trainer(model, cfg, device)
        trainer.fit(X_train_vec, y_train, X_val_vec, y_val, label_encoder=le)

        y_pred = trainer.predict(X_val_vec)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        fold_acc = float(accuracy_score(y_val, y_pred))
        fold_f1  = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        fold_results.append({"fold": fold,
                              "accuracy": round(fold_acc, 4),
                              "f1_macro": round(fold_f1, 4)})
        print(f"\n  Fold {fold} → Accuracy: {fold_acc:.4f}  F1 macro: {fold_f1:.4f}")

        # save fold checkpoint
        folder = cfg["paths"]["checkpoints"]
        os.makedirs(folder, exist_ok=True)
        torch.save(trainer.best_weights,
                   os.path.join(folder, f"{run_name}_fold{fold}.pt"))

    # ── 4. Aggregate & print ──────────────────────────────────────
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    accs = [r["accuracy"] for r in fold_results]
    f1s  = [r["f1_macro"] for r in fold_results]

    print(f"\n{'='*55}")
    print(f"  CROSS VALIDATION SUMMARY — {run_name}")
    print(f"{'='*55}")
    for r in fold_results:
        print(f"  Fold {r['fold']} → acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}")
    print(f"  {'─'*40}")
    print(f"  Mean accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean F1 macro : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    results_dir = cfg["paths"].get("results", "results/")
    evaluate(all_y_true, all_y_pred, label_encoder=le, model_name=run_name)
    plot_confusion_matrix(
        all_y_true, all_y_pred,
        label_encoder=le,
        title=f"Confusion Matrix — {run_name} ({n_folds}-Fold CV)",
        save_path=os.path.join(results_dir, f"confusion_matrix_{run_name}_cv.png"),
    )

    # ── 5. Save results + hyperparams ────────────────────────────
    save_results(cfg, fold_results, all_y_true, all_y_pred, le, run_name, results_dir)

    print("\n[main] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default=r"C:\Users\Admin\Documents\FYP\french dataset\Code\MODELS\deep learning 3\config.yaml"
    )
    args = parser.parse_args()

    # ── Start logging to file ─────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
    _vec  = _cfg["vectorizer"].upper()
    _arch = _cfg["model"]["architecture"].upper()
    _ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_dir  = _cfg["paths"].get("results", "results/")
    _log_path = os.path.join(_log_dir, f"log_{_arch}_{_vec}_{_ts}.txt")
    os.makedirs(_log_dir, exist_ok=True)
    _logger = Logger(_log_path)
    sys.stdout = _logger

    try:
        main(args.config)
    finally:
        _logger.close()
        print(f"\n[log] Full log saved → {_log_path}")