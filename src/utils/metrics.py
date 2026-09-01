import json
import os
import random
import torch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch.nn as nn

METRIC_KEYS = [
    "recall@1", "recall@5", "recall@10",
    ]


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set all global random seeds.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        print(f"  [set_seed] seed={seed}  mode=deterministic")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
        print(f"  [set_seed] seed={seed}  mode=fast (benchmark=True)")


class MetricTracker:
    def __init__(self, keys):
        self.keys   = keys
        self.sums   = {k: 0.0 for k in keys}
        self.counts = {k: 0   for k in keys}

    def update(self, metrics: dict, count: int = 1):
        for k in self.keys:
            if k in metrics:
                self.sums[k]   += metrics[k] * count
                self.counts[k] += count

    @property
    def avgs(self) -> dict:
        return {k: self.sums[k] / self.counts[k]
                for k in self.keys if self.counts[k] > 0}



def compute_metrics(img_out: torch.Tensor, txt_out: torch.Tensor) -> dict:
    B      = img_out.size(0)
    device = img_out.device

    with torch.no_grad():
        sim_matrix = img_out @ txt_out.T
        
        def recall_at_k(sim_mat, k):
            k_    = min(k, B)
            top_k = sim_mat.argsort(dim=1, descending=True)[:, :k_]
            labels = torch.arange(B, device=device).unsqueeze(1)
            return (top_k == labels).any(dim=1).float().mean().item()



    return {
        "recall@1":                recall_at_k(sim_matrix, 1),
        "recall@5":                recall_at_k(sim_matrix, 5),
        "recall@10":               recall_at_k(sim_matrix, 10),
 
    }


# Results logger & dashboard

class ResultsLogger:
    HISTORY_JSON = "history.json"
    HISTORY_CSV  = "history.csv"

    def __init__(self, metrics_dir: str, run_name: str = ""):
        self.metrics_dir = metrics_dir
        self.run_name    = run_name
        os.makedirs(metrics_dir, exist_ok=True)
        self.history: dict = {}
        path = os.path.join(metrics_dir, self.HISTORY_JSON)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.history = json.load(f)
                n = len(next(iter(self.history.values()), []))

                
                saved_run = self.history.get("_run_name", [""])[0]
                if saved_run and run_name and saved_run != run_name:
                    print(f"  [ResultsLogger] WARNING: history belongs to run "
                          f"'{saved_run}', current run is '{run_name}'. "
                          f"Starting fresh to avoid contamination.")
                    self.history = {}
                else:
                    print(f"  [ResultsLogger] Loaded existing history ({n} epochs)")
            except Exception as e:
                print(f"  [ResultsLogger] Could not load history: {e} — starting fresh")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_metrics: dict, val_metrics: dict):
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        for k, v in train_metrics.items():
            record[f"train_{k}"] = v
        for k, v in val_metrics.items():
            record[f"val_{k}"] = v
       
        record["_run_name"] = self.run_name

       
        current_len = len(next(iter(self.history.values()), []))

        for existing_key in self.history:
            if existing_key not in record:
                self.history[existing_key].append(None)

        for k, v in record.items():
            if k not in self.history:
                self.history[k] = [None] * current_len
            self.history[k].append(v)

        lengths = {k: len(v) for k, v in self.history.items()}
        if len(set(lengths.values())) > 1:
            raise RuntimeError(
                f"[ResultsLogger] History length mismatch after epoch {epoch+1}. "
                f"Lengths: { {k: l for k, l in lengths.items() if l != max(lengths.values())} }"
            )

        self._atomic_json(self.history,
                          os.path.join(self.metrics_dir, self.HISTORY_JSON))
        pd.DataFrame(self.history).to_csv(
            os.path.join(self.metrics_dir, self.HISTORY_CSV), index=False)
        self._atomic_json(record,
                          os.path.join(self.metrics_dir, f"epoch_{epoch+1:03d}.json"))
        print(f"  [ResultsLogger] Epoch {epoch+1} metrics saved")

    @staticmethod
    def _atomic_json(obj: dict, path: str):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)


def plot_training_dashboard(history: dict, epoch: int, dashboard_dir: str):
    e = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"OneEncoder Training Dashboard (Epoch {epoch})",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(e, history["train_loss"], label="train", color="#2196F3", linewidth=1.8)
    ax1.plot(e, history["val_loss"],   label="val",   color="#FF5722", linewidth=1.8)
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(alpha=0.3)


    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(e, history.get("val_recall@1",  []), label="R@1",  color="#9C27B0", linewidth=1.8)
    ax3.plot(e, history.get("val_recall@5",  []), label="R@5",  color="#3F51B5", linewidth=1.8)
    ax3.plot(e, history.get("val_recall@10", []), label="R@10", color="#00BCD4", linewidth=1.8)
    ax3.set_title("Recall@K"); ax3.set_xlabel("Epoch"); ax3.set_ylim(0, 1)
    ax3.legend(); ax3.grid(alpha=0.3)


    path = os.path.join(dashboard_dir, f"dashboard_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Dashboard] Saved  {path}")



# Checkpoints


def save_checkpoint(epoch: int, model: nn.Module, optimizer, lr_scheduler,
                    best_loss: float, num_bad_epochs: int,
                    best_epoch: int, path: str, cfg=None):
    tmp = path + ".tmp"
    payload = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "lr_scheduler":   lr_scheduler.state_dict(),
        "best_loss":      best_loss,
        "num_bad_epochs": num_bad_epochs,
        "best_epoch":     best_epoch,
    }
    
    if cfg is not None:
        payload["seed"]          = cfg.seed
        payload["deterministic"] = getattr(cfg, "deterministic", False)
        payload["batch_size"]    = cfg.batch_size
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"  [Checkpoint] Saved  {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer, lr_scheduler,
                    device, cfg=None) -> tuple:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    
    if cfg is not None:
        saved_det = ckpt.get("deterministic", None)
        if saved_det is not None and saved_det != getattr(cfg, "deterministic", False):
            print(f"  [Checkpoint] WARNING: original run used deterministic="
                  f"{saved_det}, current={cfg.deterministic}. "
                  f"Numerical differences will occur on resume.")
        saved_seed = ckpt.get("seed", None)
        if saved_seed is not None and saved_seed != cfg.seed:
            print(f"  [Checkpoint] WARNING: original seed={saved_seed}, "
                  f"current seed={cfg.seed}. Data order will differ.")
        saved_bs = ckpt.get("batch_size", None)
        if saved_bs is not None and saved_bs != cfg.batch_size:
            print(f"  [Checkpoint] WARNING: original batch_size={saved_bs}, "
                  f"current={cfg.batch_size}. LR schedule effective steps differ.")

    print(f"  [Checkpoint] Loaded epoch {ckpt['epoch']+1}  "
          f"best_loss={ckpt['best_loss']:.4f}")
    return (ckpt["epoch"] + 1, ckpt["best_loss"],
            ckpt["num_bad_epochs"], ckpt["best_epoch"])
