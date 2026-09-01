import os
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast


DEFAULT_SEED = 42
VAL_FRACTION_FROM_TRAIN = 0.1


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True



@dataclass
class FTBackboneAdapter:
    fresh_backbone: Callable[[], nn.Module]
    extract_fn:     Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
    tokenize_fn:    Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]]
    backbone_params: Callable[[nn.Module], Iterable[nn.Parameter]]
    name:           str = "backbone"


def discover_feature_dim(adapter: FTBackboneAdapter, device: torch.device) -> int:
    bb = adapter.fresh_backbone().to(device)
    bb.eval()
    ids, mask = adapter.tokenize_fn(["a sanity sentence", "another one here"])
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        feats = adapter.extract_fn(bb, ids, mask)
    if feats.dim() != 2:
        raise RuntimeError(f"[{adapter.name}] extract_fn must return (B,D); got {tuple(feats.shape)}")
    del bb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats.shape[-1]



def build_head(in_dim: int, num_classes: int) -> nn.Module:

    return nn.Linear(in_dim, num_classes)

class FineTunedClassifier(nn.Module):
    def __init__(self, adapter: FTBackboneAdapter, backbone: nn.Module,
                 in_dim: int, num_classes: int, device: torch.device):
        super().__init__()
        self.adapter    = adapter
        self.backbone   = backbone
        self.classifier = build_head(in_dim, num_classes).to(device)
        self.device     = device

    def forward(self, input_ids, attention_mask):
        feats = self.adapter.extract_fn(self.backbone,
                                        input_ids.to(self.device),
                                        attention_mask.to(self.device))
        return self.classifier(feats)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.eval()
        correct = total = 0
        for ids, mask, labels in tqdm(loader, desc="  eval", leave=False):
            labels = labels.to(self.device, non_blocking=True)
            logits = self(ids, mask)
            total  += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
        return {"top1": correct / max(total, 1)}


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = list(texts); self.labels = list(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], int(self.labels[idx])


def make_collate(tokenize_fn):
    def _collate(batch):
        texts, labels = zip(*batch)
        ids, mask = tokenize_fn(list(texts))
        return ids, mask, torch.tensor(labels, dtype=torch.long)
    return _collate


DATASET_ALIASES = {"emotion": "emotion", "sst2": "sst2", "sst-2": "sst2", "trec": "trec"}
DATASET_CHOICES = ["emotion", "sst2", "trec"]


def canonicalize_dataset_name(name: str) -> str:
    key = name.lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {DATASET_CHOICES}")
    return DATASET_ALIASES[key]


def _loader(ds, bs, shuffle, nw, collate):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      collate_fn=collate, pin_memory=torch.cuda.is_available(),
                      persistent_workers=(nw > 0))


def build_loaders(dataset_name, batch_size, num_workers, collate, seed: int = DEFAULT_SEED):
    """
    Returns (train_loader, val_loader, test_loader, num_classes).

    """
    from datasets import load_dataset as hf
    dataset_name = canonicalize_dataset_name(dataset_name)
    generator = torch.Generator().manual_seed(seed)
    def ds(t, l): return TextDataset(list(t), list(l))

    if dataset_name == "emotion":
        d = hf("dair-ai/emotion")
        return (_loader(ds(d["train"]["text"], d["train"]["label"]), batch_size, True, num_workers, collate),
                _loader(ds(d["validation"]["text"], d["validation"]["label"]), batch_size, False, num_workers, collate),
                _loader(ds(d["test"]["text"], d["test"]["label"]), batch_size, False, num_workers, collate),
                6)
    if dataset_name == "sst2":
        d = hf("stanfordnlp/sst2")
        train_full = ds(d["train"]["sentence"], d["train"]["label"])
        te         = ds(d["validation"]["sentence"], d["validation"]["label"]) 
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(train_full))
        train_n = len(train_full) - val_n
        tr_sub, val_sub = random_split(train_full, [train_n, val_n], generator=generator)
        return (_loader(tr_sub,  batch_size, True,  num_workers, collate),
                _loader(val_sub, batch_size, False, num_workers, collate),
                _loader(te,      batch_size, False, num_workers, collate),
                2)
    if dataset_name == "trec":
        d = hf("CogComp/trec", trust_remote_code=True)
        train_full = ds(d["train"]["text"], d["train"]["coarse_label"])
        te         = ds(d["test"]["text"],  d["test"]["coarse_label"])

        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(train_full))
        train_n = len(train_full) - val_n
        tr_sub, val_sub = random_split(train_full, [train_n, val_n], generator=generator)

        return (_loader(tr_sub,  batch_size, True,  num_workers, collate),
                _loader(val_sub, batch_size, False, num_workers, collate),
                _loader(te,      batch_size, False, num_workers, collate),
                6)
    raise AssertionError("Unreachable")



def build_optimizer_and_scheduler(model: FineTunedClassifier, *,
                                   lr: float, backbone_lr_scale: float,
                                   weight_decay: float, warmup_ratio: float,
                                   num_training_steps: int):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")

    bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
    bb_param_ids = {id(p) for p in model.adapter.backbone_params(model.backbone)}

    for name, p in model.backbone.named_parameters():
        if id(p) not in bb_param_ids:
            continue
        (bb_nodecay if any(nd in name for nd in no_decay) else bb_decay).append(p)
    for name, p in model.classifier.named_parameters():
        (hd_nodecay if any(nd in name for nd in no_decay) else hd_decay).append(p)

    bb_lr = lr * backbone_lr_scale
    groups = [
        {"params": bb_decay,   "lr": bb_lr, "weight_decay": weight_decay},
        {"params": bb_nodecay, "lr": bb_lr, "weight_decay": 0.0},
        {"params": hd_decay,   "lr": lr,    "weight_decay": weight_decay},
        {"params": hd_nodecay, "lr": lr,    "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups)

    warmup = int(warmup_ratio * num_training_steps)
    def lr_lambda(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        prog = float(step - warmup) / max(1, num_training_steps - warmup)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def set_backbone_frozen(model: FineTunedClassifier, frozen: bool) -> None:
    """Toggle requires_grad on the backbone-group params only."""
    bb_param_ids = {id(p) for p in model.adapter.backbone_params(model.backbone)}
    for p in model.backbone.parameters():
        if id(p) in bb_param_ids:
            p.requires_grad = not frozen



def train_finetune(adapter: FTBackboneAdapter, in_dim: int, dataset_name: str, *,
                   epochs: int, patience: int, freeze_epochs: int,
                   batch_size: int, lr: float, backbone_lr_scale: float,
                   weight_decay: float, label_smoothing: float,
                   warmup_ratio: float, max_grad_norm: float, use_amp: bool,
                   num_workers: int, device: torch.device, out_dir: str,
                   seed: int = DEFAULT_SEED) -> Dict:
    dataset_name = canonicalize_dataset_name(dataset_name)
    print(f"\n{'=' * 64}\n  Dataset : {dataset_name.upper()}  "
          f"|  gradual unfreeze @ epoch {freeze_epochs+1}\n{'=' * 64}")

    collate = make_collate(adapter.tokenize_fn)
    train_loader, val_loader, test_loader, num_classes = build_loaders(
        dataset_name, batch_size, num_workers, collate, seed=seed)
    train_n = len(train_loader.dataset)
    val_n   = len(val_loader.dataset)
    test_n  = len(test_loader.dataset)
    print(f"  Split   : train={train_n}  val={val_n}  test={test_n}  classes={num_classes}")

    backbone = adapter.fresh_backbone().to(device)   
    model    = FineTunedClassifier(adapter, backbone, in_dim, num_classes, device).to(device)

    num_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=lr, backbone_lr_scale=backbone_lr_scale,
        weight_decay=weight_decay, warmup_ratio=warmup_ratio,
        num_training_steps=num_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler    = GradScaler(enabled=use_amp)

    best_val, best_epoch, bad, executed = -1.0, 0, 0, 0
    best_path = os.path.join(out_dir, f"best_{adapter.name}_{dataset_name}.pt")

    for epoch in range(epochs):
        executed = epoch + 1
        frozen = epoch < freeze_epochs
        set_backbone_frozen(model, frozen)
        if epoch == freeze_epochs and freeze_epochs > 0:
            print(f"  [Unfreeze] Backbone unlocked at epoch {epoch+1}; resetting early-stop counter.")
            bad = 0                                    
            best_val = -1.0                           

        model.train()
        running = correct = total = 0.0
        for ids, mask, labels in tqdm(train_loader, desc=f"  ep {epoch+1:3d}/{epochs}", leave=False):
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(ids, mask)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        train_acc = correct / max(total, 1)
        val = model.evaluate(val_loader)
        phase = "frozen" if frozen else "ft"
        print(f"  ep {epoch+1:3d} [{phase}] | loss {running/max(len(train_loader),1):.4f}"
              f" | train {train_acc:.4f} | val {val['top1']:.4f}"
              f" | lr_bb {optimizer.param_groups[0]['lr']:.1e}"
              f" | lr_hd {optimizer.param_groups[2]['lr']:.1e}")

        if val["top1"] > best_val:
            best_val, best_epoch, bad = val["top1"], epoch + 1, 0
            torch.save({"backbone": model.backbone.state_dict(),
                        "classifier": model.classifier.state_dict()}, best_path)
            print(f"    -> new best val_acc={best_val:.4f}")
        else:
            bad += 1
            if (epoch >= freeze_epochs) and bad >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.classifier.load_state_dict(ckpt["classifier"])
    test_top1 = round(model.evaluate(test_loader)["top1"], 4)
    print(f"\n  {dataset_name.upper()} best epoch {best_epoch} | test_acc {test_top1:.4f}")

    del model, backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"dataset": dataset_name, "num_classes": num_classes,
            "train_size": train_n, "val_size": val_n, "test_size": test_n,
            "best_val_top1": round(best_val, 4),
            "test_top1": test_top1,
            "best_epoch": best_epoch, "total_epochs": executed}



def _fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def print_results_table(results, backbone_name, freeze_epochs, backbone_lr_scale):
    print(f"\n\n{'=' * 72}")
    print(f"  Text Fine-Tuning Summary  [{backbone_name}]")
    print(f"  gradual unfreeze @ {freeze_epochs}  |  backbone_lr_scale={backbone_lr_scale}")
    print(f"{'=' * 72}\n")
    print("| Dataset  | Classes | Val Acc   | Test Acc  | Best Epoch |")
    print("|----------|---------|-----------|-----------|------------|")
    for r in results:
        print(f"| {str(r['dataset']):<8} | {str(r['num_classes']):<7} "
              f"| {_fmt(r['best_val_top1']):<9} | {_fmt(r.get('test_top1','N/A')):<9} "
              f"| {str(r['best_epoch']):<10} |")
    finite_val = [r['best_val_top1'] for r in results if isinstance(r['best_val_top1'], (int, float))]
    finite_test = [r['test_top1'] for r in results if isinstance(r.get('test_top1'), (int, float))]
    if finite_val:
        print(f"\n  Mean Val Acc  : {sum(finite_val)/len(finite_val):.4f}")
    if finite_test:
        print(f"  Mean Test Acc : {sum(finite_test)/len(finite_test):.4f}")
    print()


def save_results(results, out_dir, backbone_name, model_path, cfg_meta):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"backbone": backbone_name, "model_path": model_path,
            "protocol": "fixed", "freeze_strategy": "gradual",
            "timestamp": ts, **cfg_meta, "results": results}
    with open(os.path.join(out_dir, f"finetune_{backbone_name}_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, f"finetune_{backbone_name}_{ts}.csv"), index=False)
    print(f"  Results saved to {out_dir}")



def run_finetune(adapter: FTBackboneAdapter, datasets: List[str], out_dir: str, *,
                 epochs: int = 200, patience: int = 200, freeze_epochs: int = 0,
                 batch_size: int = 64, lr: float = 4e-3, backbone_lr_scale: float = 0.1,
                 weight_decay: float = 0.0001, label_smoothing: float = 0.1,
                 warmup_ratio: float = 0.06, max_grad_norm: float = 1.0,
                 use_amp: bool = True, num_workers: int = 4,
                 device: torch.device = None, seed: int = DEFAULT_SEED,
                 model_path: str = "") -> None:
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed, deterministic=True)

    datasets = (DATASET_CHOICES if not datasets or datasets == ["all"]
                else [canonicalize_dataset_name(d) for d in datasets])

    in_dim = discover_feature_dim(adapter, device)
    os.makedirs(out_dir, exist_ok=True)

    cfg_meta = {"epochs": epochs, "patience": patience, "freeze_epochs": freeze_epochs,
                "batch_size": batch_size, "lr": lr, "backbone_lr_scale": backbone_lr_scale,
                "weight_decay": weight_decay, "label_smoothing": label_smoothing,
                "warmup_ratio": warmup_ratio, "max_grad_norm": max_grad_norm,
                "feature_dim": in_dim}

    print(f"\n{'=' * 64}")
    print(f"  Text FULL fine-tuning  |  backbone={adapter.name}")
    print(f"  feature dim={in_dim}  | fixed protocol | gradual unfreeze @ {freeze_epochs}")
    print(f"  lr={lr}  backbone_lr_scale={backbone_lr_scale}  wd={weight_decay}")
    print(f"  datasets={datasets}")
    print(f"{'=' * 64}")

    results = []
    for ds in datasets:
        try:
            set_seed(seed, deterministic=True)
            results.append(train_finetune(
                adapter, in_dim, ds, epochs=epochs, patience=patience,
                freeze_epochs=freeze_epochs, batch_size=batch_size, lr=lr,
                backbone_lr_scale=backbone_lr_scale, weight_decay=weight_decay,
                label_smoothing=label_smoothing, warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm, use_amp=use_amp,
                num_workers=num_workers, device=device, out_dir=out_dir, seed=seed))
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {ds} failed:\n{traceback.format_exc()}")
            results.append({"dataset": ds, "num_classes": "_", "train_size": "_",
                            "val_size": "_", "test_size": "_",
                            "best_val_top1": "ERROR", "test_top1": "ERROR",
                            "best_epoch": "_", "total_epochs": "_", "error": str(e)})

    print_results_table(results, adapter.name, freeze_epochs, backbone_lr_scale)
    save_results(results, out_dir, adapter.name, model_path, cfg_meta)