import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics import evaluate, plot_confusion_matrix, epoch_log


class Trainer:

    def __init__(self, model, cfg, device):
        self.model  = model
        self.cfg    = cfg
        self.device = device

        t = cfg["training"]
        self.epochs     = t["epochs"]
        self.batch_size = t["batch_size"]
        self.patience   = t["patience"]
        self.clip_grad  = t.get("clip_grad_norm", 1.0)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(),
                              lr=t["learning_rate"],
                              weight_decay=t.get("weight_decay", 1e-5))
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=t.get("scheduler_factor", 0.5),
            patience=t.get("scheduler_patience", 3),
        )

        self.best_loss    = float("inf")
        self.best_weights = None
        self.counter      = 0

    def _loader(self, X, y, shuffle=False):
        X_t = torch.tensor(X, dtype=torch.float32)   # float32 for vectorizer input
        y_t = torch.tensor(np.array(y), dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t),
                          batch_size=self.batch_size, shuffle=shuffle)

    def _epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if train:
                self.optimizer.zero_grad()

            logits = self.model(X_batch)
            loss   = self.criterion(logits, y_batch)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            preds       = logits.argmax(dim=1)
            total_loss += loss.item() * len(y_batch)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

        return total_loss / total, correct / total

    def fit(self, X_train, y_train, X_val, y_val, label_encoder=None):
        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader   = self._loader(X_val,   y_val,   shuffle=False)

        arch = self.cfg["model"]["architecture"].upper()
        vec  = self.cfg["vectorizer"].upper()

        print(f"\n{'='*55}")
        print(f"  {arch} + {vec}")
        print(f"  Epochs: {self.epochs}  |  Batch: {self.batch_size}  |  Patience: {self.patience}")
        print(f"  LR: {self.cfg['training']['learning_rate']}  |  Clip: {self.clip_grad}")
        print(f"{'='*55}\n")

        for epoch in range(1, self.epochs + 1):
            train_loss, _       = self._epoch(train_loader, train=True)
            val_loss,   val_acc = self._epoch(val_loader,   train=False)

            epoch_log(epoch, self.epochs, train_loss, val_loss, val_acc)
            self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                self.best_loss    = val_loss
                self.counter      = 0
                self.best_weights = {k: v.clone()
                                     for k, v in self.model.state_dict().items()}
                self._save_checkpoint()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"\n[Trainer] Early stopping at epoch {epoch}.")
                    break

        if self.best_weights:
            self.model.load_state_dict(self.best_weights)
            print("[Trainer] Best weights restored.")

        print("\n[Trainer] Final evaluation:\n")
        y_pred      = self.predict(X_val)
        run_name    = f"{arch}_{vec}"
        results_dir = self.cfg["paths"].get("results", "results/")
        os.makedirs(results_dir, exist_ok=True)
        save_path   = os.path.join(results_dir, f"confusion_matrix_{run_name}.png")

        evaluate(y_val, y_pred, label_encoder=label_encoder, model_name=run_name)
        plot_confusion_matrix(y_val, y_pred, label_encoder=label_encoder,
                              title=f"Confusion Matrix — {run_name}",
                              save_path=save_path)

    def predict(self, X):
        self.model.eval()
        loader = self._loader(X, np.zeros(len(X), dtype=int), shuffle=False)
        preds  = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                preds.extend(self.model(X_batch).argmax(dim=1).cpu().numpy())
        return np.array(preds)

    def _save_checkpoint(self):
        folder = self.cfg["paths"]["checkpoints"]
        os.makedirs(folder, exist_ok=True)
        arch = self.cfg["model"]["architecture"]
        vec  = self.cfg["vectorizer"]
        path = os.path.join(folder, f"{arch}_{vec}.pt")
        torch.save(self.model.state_dict(), path)
        print(f"  [Trainer] Checkpoint → {path}")
