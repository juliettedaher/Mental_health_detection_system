import argparse
import os
import yaml
import torch

from dataset import load_french_dataset
from lstm    import get_model
from trainer import Trainer


def main(config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vec_choice = cfg["vectorizer"].lower()
    arch       = cfg["model"]["architecture"].upper()

    print("\n" + "=" * 60)
    print("  French Mental Health Classifier")
    print(f"  Model      : {arch}")
    print(f"  Vectorizer : {vec_choice.upper()}")
    print(f"  Input size : {cfg['model']['input_size']}")
    print("=" * 60 + "\n")

    # ── 1. Load dataset ───────────────────────────────────────────
    X_train, X_test, y_train, y_test, label_encoder = load_french_dataset(
        path         = cfg["paths"]["data"],
        text_col     = cfg["dataset"]["text_col"],
        label_col    = cfg["dataset"]["label_col"],
        test_size    = cfg["dataset"]["test_size"],
        random_state = cfg["dataset"]["random_state"],
    )

    # ── 2. Vectorize ──────────────────────────────────────────────
    if vec_choice == "tfidf":
        from tfidf import TfidfVectorizerWrapper

        tfidf_path = cfg["paths"]["tfidf_vectorizer"]

        if os.path.exists(tfidf_path):
            print("[main] Loading saved TF-IDF vectorizer...")
            vec         = TfidfVectorizerWrapper.load(tfidf_path)
            X_train_vec = vec.transform(X_train)
            X_test_vec  = vec.transform(X_test)
        else:
            print("[main] Fitting TF-IDF vectorizer on training data...")
            os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)
            vec         = TfidfVectorizerWrapper(max_features=cfg["model"]["input_size"])
            X_train_vec = vec.fit_transform(X_train)
            X_test_vec  = vec.transform(X_test)
            vec.save(tfidf_path)

    elif vec_choice == "camembert":
        from camembert import CamembertVectorizer

        train_path = cfg["paths"]["cam_train_emb"]
        test_path  = cfg["paths"]["cam_test_emb"]

        if os.path.exists(train_path) and os.path.exists(test_path):
            print("[main] Loading saved CamemBERT embeddings...")
            vec         = CamembertVectorizer()
            X_train_vec = vec.load_embeddings(train_path)
            X_test_vec  = vec.load_embeddings(test_path)
        else:
            print("[main] Encoding with CamemBERT (~8 min on CPU)...")
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            vec         = CamembertVectorizer()
            X_train_vec = vec.encode_dataset(X_train)
            X_test_vec  = vec.encode_dataset(X_test)
            vec.save_embeddings(X_train_vec, train_path)
            vec.save_embeddings(X_test_vec,  test_path)
            print("[main] Embeddings saved — next run loads from disk.")

    else:
        raise ValueError(f"Unknown vectorizer '{vec_choice}'. Use 'tfidf' or 'camembert'.")

    print(f"[main] X_train: {X_train_vec.shape} | X_test: {X_test_vec.shape}\n")

    # ── 3. Build model ────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device: {device}")

    model        = get_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Parameters: {total_params:,}\n")

    # ── 4. Train ──────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device)
    trainer.fit(X_train_vec, y_train, X_test_vec, y_test,
                label_encoder=label_encoder)

    print("\n[main] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default=r"C:\Users\Admin\Documents\FYP\french dataset\Code\MODELS\deep learning lstm\config.yaml"
    )
    args = parser.parse_args()
    main(args.config)
