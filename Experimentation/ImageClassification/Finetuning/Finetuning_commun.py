import os
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2


DEFAULT_SEED = 42

VAL_FRACTION_FROM_TRAIN = 0.1     # CIFAR-10/100, Oxford-Pets -> 90/10 train/val
# Caltech101 
CALTECH_TRAIN_FRACTION = 0.8
CALTECH_VAL_FRACTION   = 0.1
CALTECH_TEST_FRACTION  = 0.1     
# Tiny-ImageNet ships train + 'val' (no labeled test).


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
class FTImageBackboneAdapter:
    fresh_backbone:  Callable[[], nn.Module]
    extract_fn:      Callable[[nn.Module, torch.Tensor], torch.Tensor]
    norm_mean:       Tuple[float, float, float]
    norm_std:        Tuple[float, float, float]
    backbone_params: Callable[[nn.Module], Iterable[nn.Parameter]]
    name:            str = "backbone"


def discover_feature_dim(adapter: FTImageBackboneAdapter, device: torch.device,
                         image_size: int = 224) -> int:
    bb = adapter.fresh_backbone().to(device)
    bb.eval()
    dummy = torch.zeros(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        feats = adapter.extract_fn(bb, dummy)
    if feats.dim() != 2:
        raise RuntimeError(f"[{adapter.name}] extract_fn must return (B,D); got {tuple(feats.shape)}")
    del bb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats.shape[-1]



def get_train_transform(image_size, mean, std) -> A.Compose:
    
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_eval_transform(image_size, mean, std) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])



class AlbumentationsWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset; self.transform = transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img.convert("RGB"))
        return self.transform(image=img)["image"], int(label)


class TinyImageNetDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root) / "tiny-imagenet-200"
        self.split = split; self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        if not self.root.exists():
            raise FileNotFoundError(f"Tiny ImageNet root not found: {self.root}")
        wnids = [l.strip() for l in (self.root / "wnids.txt").read_text().splitlines() if l.strip()]
        self.class_to_idx = {w: i for i, w in enumerate(sorted(wnids))}
        if split == "train":
            for wnid, idx in self.class_to_idx.items():
                d = self.root / "train" / wnid / "images"
                if d.is_dir():
                    for p in sorted(d.glob("*.JPEG")):
                        self.samples.append((str(p), idx))
        else:
            ann = self.root / "val" / "val_annotations.txt"
            for line in ann.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1] in self.class_to_idx:
                    p = self.root / "val" / "images" / parts[0]
                    if p.exists():
                        self.samples.append((str(p), self.class_to_idx[parts[1]]))
        if not self.samples:
            raise RuntimeError(f"No Tiny ImageNet samples for split='{split}'")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, int(label)


DATASET_ALIASES = {
    "caltech101": "caltech101", "cifar10": "cifar10", "cifar100": "cifar100",
    "oxford_pets": "oxford_pets", "oxfordpets": "oxford_pets",
    "tiny_imagenet": "tiny_imagenet", "tinyimagenet200": "tiny_imagenet",
    "tinyimagenet": "tiny_imagenet",
}
DATASET_CHOICES = ["caltech101", "cifar10", "cifar100", "oxford_pets", "tiny_imagenet"]


def canonicalize_dataset_name(name):
    key = name.lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {DATASET_CHOICES}")
    return DATASET_ALIASES[key]


def _loader(ds, bs, shuffle, nw):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=(nw > 0))


def build_loaders(dataset_name, data_root, train_tf, eval_tf,
                  batch_size, num_workers, seed=DEFAULT_SEED):
    """
    Returns (train_loader, val_loader, test_loader, num_classes).

    """
    dataset_name = canonicalize_dataset_name(dataset_name)
    g = torch.Generator().manual_seed(seed)

    if dataset_name == "cifar10":
        tr_raw = torchvision.datasets.CIFAR10(data_root, train=True, download=True)
        te_raw = torchvision.datasets.CIFAR10(data_root, train=False, download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_raw))
        train_n = len(tr_raw) - val_n
        tr_sub, val_sub = random_split(tr_raw, [train_n, val_n], generator=g)
        return (_loader(AlbumentationsWrapper(tr_sub,  train_tf), batch_size, True,  num_workers),
                _loader(AlbumentationsWrapper(val_sub, eval_tf),  batch_size, False, num_workers),
                _loader(AlbumentationsWrapper(te_raw,  eval_tf),  batch_size, False, num_workers),
                10)

    if dataset_name == "cifar100":
        tr_raw = torchvision.datasets.CIFAR100(data_root, train=True, download=True)
        te_raw = torchvision.datasets.CIFAR100(data_root, train=False, download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_raw))
        train_n = len(tr_raw) - val_n
        tr_sub, val_sub = random_split(tr_raw, [train_n, val_n], generator=g)
        return (_loader(AlbumentationsWrapper(tr_sub,  train_tf), batch_size, True,  num_workers),
                _loader(AlbumentationsWrapper(val_sub, eval_tf),  batch_size, False, num_workers),
                _loader(AlbumentationsWrapper(te_raw,  eval_tf),  batch_size, False, num_workers),
                100)

    if dataset_name == "oxford_pets":
        tr_raw = torchvision.datasets.OxfordIIITPet(data_root, split="trainval", download=True)
        te_raw = torchvision.datasets.OxfordIIITPet(data_root, split="test", download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_raw))
        train_n = len(tr_raw) - val_n
        tr_sub, val_sub = random_split(tr_raw, [train_n, val_n], generator=g)
        return (_loader(AlbumentationsWrapper(tr_sub,  train_tf), batch_size, True,  num_workers),
                _loader(AlbumentationsWrapper(val_sub, eval_tf),  batch_size, False, num_workers),
                _loader(AlbumentationsWrapper(te_raw,  eval_tf),  batch_size, False, num_workers),
                37)

    if dataset_name == "caltech101":
        raw = torchvision.datasets.Caltech101(data_root, download=True)
        n       = len(raw)
        train_n = int(CALTECH_TRAIN_FRACTION * n)
        val_n   = int(CALTECH_VAL_FRACTION * n)
        test_n  = n - train_n - val_n         
        tr_sub, val_sub, te_sub = random_split(raw, [train_n, val_n, test_n], generator=g)
        return (_loader(AlbumentationsWrapper(tr_sub,  train_tf), batch_size, True,  num_workers),
                _loader(AlbumentationsWrapper(val_sub, eval_tf),  batch_size, False, num_workers),
                _loader(AlbumentationsWrapper(te_sub,  eval_tf),  batch_size, False, num_workers),
                101)

    if dataset_name == "tiny_imagenet":
        tr_full = TinyImageNetDataset(data_root, "train", train_tf)
        te      = TinyImageNetDataset(data_root, "val",   eval_tf)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_full))
        train_n = len(tr_full) - val_n
        tr_sub, val_sub = random_split(tr_full, [train_n, val_n], generator=g)
        return (_loader(tr_sub,  batch_size, True,  num_workers),
                _loader(val_sub, batch_size, False, num_workers),
                _loader(te,      batch_size, False, num_workers),
                200)

    raise AssertionError("Unreachable")


def build_head(in_dim, num_classes):
    hidden = max(512, in_dim)
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden // 2, num_classes),
    )



class FineTunedClassifier(nn.Module):
    def __init__(self, adapter, backbone, in_dim, num_classes, device):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.classifier = build_head(in_dim, num_classes).to(device)
        self.device = device

    def forward(self, images):
        feats = self.adapter.extract_fn(self.backbone, images.to(self.device))
        return self.classifier(feats)

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval()
        correct = total = 0
        for images, labels in tqdm(loader, desc="  eval", leave=False):
            labels = labels.to(self.device, non_blocking=True)
            logits = self(images)
            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
        return {"top1": correct / max(total, 1)}



def build_optimizer_and_scheduler(model, *, lr, backbone_lr_scale, weight_decay,
                                   warmup_ratio, num_training_steps):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
    bb_ids = {id(p) for p in model.adapter.backbone_params(model.backbone)}
    for name, p in model.backbone.named_parameters():
        if id(p) not in bb_ids:
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
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_backbone_frozen(model, frozen):
    bb_ids = {id(p) for p in model.adapter.backbone_params(model.backbone)}
    for p in model.backbone.parameters():
        if id(p) in bb_ids:
            p.requires_grad = not frozen



def train_finetune(adapter, in_dim, dataset_name, data_root, *,
                   epochs, patience, freeze_epochs, batch_size, lr,
                   backbone_lr_scale, weight_decay, label_smoothing,
                   warmup_ratio, max_grad_norm, use_amp, num_workers,
                   image_size, device, out_dir, seed=DEFAULT_SEED):
    dataset_name = canonicalize_dataset_name(dataset_name)
    print(f"\n{'=' * 64}\n  Dataset : {dataset_name.upper()}  "
          f"|  gradual unfreeze @ epoch {freeze_epochs+1}\n{'=' * 64}")

    train_tf = get_train_transform(image_size, adapter.norm_mean, adapter.norm_std)
    eval_tf  = get_eval_transform(image_size, adapter.norm_mean, adapter.norm_std)
    train_loader, val_loader, test_loader, num_classes = build_loaders(
        dataset_name, data_root, train_tf, eval_tf, batch_size, num_workers, seed=seed)
    train_n = len(train_loader.dataset)
    val_n   = len(val_loader.dataset)
    test_n  = len(test_loader.dataset)
    print(f"  Split   : train={train_n}  val={val_n}  test={test_n}  classes={num_classes}")

    backbone = adapter.fresh_backbone().to(device)
    model    = FineTunedClassifier(adapter, backbone, in_dim, num_classes, device).to(device)

    num_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=lr, backbone_lr_scale=backbone_lr_scale, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio, num_training_steps=num_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler    = GradScaler(enabled=use_amp)

    best_val, best_epoch, bad, executed = -1.0, 0, 0, 0
    best_path = os.path.join(out_dir, f"best_{adapter.name}_{dataset_name}.pt")

    for epoch in range(epochs):
        executed = epoch + 1
        frozen = epoch < freeze_epochs
        set_backbone_frozen(model, frozen)
        if epoch == freeze_epochs and freeze_epochs > 0:
            print(f"  [Unfreeze] Backbone unlocked at epoch {epoch+1}; resetting early-stop.")
            bad = 0; best_val = -1.0

        model.train()
        running = correct = total = 0.0
        for images, labels in tqdm(train_loader, desc=f"  ep {epoch+1:3d}/{epochs}", leave=False):
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

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

    
    final_val = round(model.evaluate(val_loader)["top1"], 4)
    test_top1 = round(model.evaluate(test_loader)["top1"], 4)
    print(f"\n  {dataset_name.upper()} best epoch {best_epoch} | test_acc {test_top1:.4f}")

    del model, backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"dataset": dataset_name, "num_classes": num_classes,
            "train_size": train_n, "val_size": val_n, "test_size": test_n,
            "best_val_top1": round(best_val, 4), "reloaded_val_top1": final_val,
            "test_top1": test_top1,
            "best_epoch": best_epoch, "total_epochs": executed}



def _fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def print_results_table(results, backbone_name, freeze_epochs, backbone_lr_scale):
    print(f"\n\n{'=' * 72}")
    print(f"  Image Fine-Tuning Summary  [{backbone_name}]")
    print(f"  gradual unfreeze @ {freeze_epochs}  |  backbone_lr_scale={backbone_lr_scale}")
    print(f"{'=' * 72}\n")
    print("| Dataset       | Classes | Val Top-1 | Test Top-1 | Best Epoch |")
    print("|---------------|---------|-----------|------------|------------|")
    for r in results:
        print(f"| {str(r['dataset']):<13} | {str(r['num_classes']):<7} "
              f"| {_fmt(r['best_val_top1']):<9} | {_fmt(r.get('test_top1', 'N/A')):<10} "
              f"| {str(r['best_epoch']):<10} |")
    finite_val = [r['best_val_top1'] for r in results if isinstance(r['best_val_top1'], (int, float))]
    finite_test = [r['test_top1'] for r in results if isinstance(r.get('test_top1'), (int, float))]
    if finite_val:
        print(f"\n  Mean Val Top-1  : {sum(finite_val)/len(finite_val):.4f}")
    if finite_test:
        print(f"  Mean Test Top-1 : {sum(finite_test)/len(finite_test):.4f}")
    print()


def save_results(results, out_dir, backbone_name, model_path, cfg_meta):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"backbone": backbone_name, "model_path": model_path,
            "protocol": "fixed", "freeze_strategy": "gradual",
            "timestamp": ts, **cfg_meta, "results": results}
    with open(os.path.join(out_dir, f"finetune_img_{backbone_name}_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, f"finetune_img_{backbone_name}_{ts}.csv"), index=False)
    print(f"  Results saved to {out_dir}")



def run_finetune(adapter, datasets, data_root, out_dir, *,
                 epochs=100, patience=50, freeze_epochs=0, batch_size=64,
                 lr=1e-3, backbone_lr_scale=0.1, weight_decay=0.0001,
                 label_smoothing=0.1, warmup_ratio=0.06, max_grad_norm=1.0,
                 use_amp=True, num_workers=4, image_size=224,
                 device=None, seed=DEFAULT_SEED, model_path=""):
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed, deterministic=True)
    datasets = (DATASET_CHOICES if not datasets or datasets == ["all"]
                else [canonicalize_dataset_name(d) for d in datasets])

    in_dim = discover_feature_dim(adapter, device, image_size)
    os.makedirs(out_dir, exist_ok=True)

    cfg_meta = {"epochs": epochs, "patience": patience, "freeze_epochs": freeze_epochs,
                "batch_size": batch_size, "lr": lr, "backbone_lr_scale": backbone_lr_scale,
                "weight_decay": weight_decay, "label_smoothing": label_smoothing,
                "warmup_ratio": warmup_ratio, "max_grad_norm": max_grad_norm,
                "image_size": image_size, "feature_dim": in_dim,
                "augmentation": "none"}

    print(f"\n{'=' * 64}")
    print(f"  Image FULL fine-tuning  |  backbone={adapter.name}")
    print(f"  feature dim={in_dim} | fixed protocol | gradual unfreeze @ {freeze_epochs}")
    print(f"  lr={lr} backbone_lr_scale={backbone_lr_scale} wd={weight_decay} | aug=none")
    print(f"  norm mean={adapter.norm_mean} std={adapter.norm_std}")
    print(f"  datasets={datasets}")
    print(f"{'=' * 64}")

    results = []
    for ds in datasets:
        try:
            set_seed(seed, deterministic=True)
            results.append(train_finetune(
                adapter, in_dim, ds, data_root, epochs=epochs, patience=patience,
                freeze_epochs=freeze_epochs, batch_size=batch_size, lr=lr,
                backbone_lr_scale=backbone_lr_scale, weight_decay=weight_decay,
                label_smoothing=label_smoothing, warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm, use_amp=use_amp, num_workers=num_workers,
                image_size=image_size, device=device, out_dir=out_dir, seed=seed))
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {ds} failed:\n{traceback.format_exc()}")
            results.append({"dataset": ds, "num_classes": "_", "train_size": "_",
                            "val_size": "_", "test_size": "_", "best_val_top1": "ERROR",
                            "test_top1": "ERROR",
                            "best_epoch": "_", "total_epochs": "_", "error": str(e)})

    print_results_table(results, adapter.name, freeze_epochs, backbone_lr_scale)
    save_results(results, out_dir, adapter.name, model_path, cfg_meta)