import os
import json
import random
import types
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2




HEAD_HIDDEN   = 512        
HEAD_DROPOUT  = 0.5
SCHED_ETA_MIN = 1e-5
DEFAULT_SEED  = 42


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
class BackboneAdapter:
    """
    Wraps a frozen backbone for probing.

    """
    module:     nn.Module
    extract_fn: Callable[[torch.Tensor], torch.Tensor]
    norm_mean:  Tuple[float, float, float]
    norm_std:   Tuple[float, float, float]
    name:       str = "backbone"

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_fn(images)


def discover_feature_dim(adapter: BackboneAdapter, device: torch.device,
                         image_size: int = 224) -> int:
    adapter.module.eval()
    dummy = torch.zeros(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        feats = adapter.extract_fn(dummy)
    if feats.dim() != 2:
        raise RuntimeError(
            f"[{adapter.name}] extract_fn must return (B, D); got shape "
            f"{tuple(feats.shape)}. Pool/flatten inside the adapter."
        )
    return feats.shape[-1]



def get_transform(image_size: int,
                  mean: Tuple[float, float, float],
                  std:  Tuple[float, float, float]) -> A.Compose:

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])



class AlbumentationsWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform: A.Compose):
        self.dataset   = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        img = np.array(img.convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, int(label)


class TinyImageNetDataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 transform: A.Compose = None):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.root      = Path(root) / "tiny-imagenet-200"
        self.split     = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"Tiny ImageNet root not found: {self.root}\n"
                f"Expected dataset under: <data_root>/tiny-imagenet-200/"
            )
        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet file: {wnids_path}")

        with open(wnids_path, "r", encoding="utf-8") as f:
            wnids = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {wnid: i for i, wnid in enumerate(sorted(wnids))}

        if split == "train":
            for wnid, idx in self.class_to_idx.items():
                img_dir = self.root / "train" / wnid / "images"
                if not img_dir.exists():
                    continue
                for img_path in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append((str(img_path), idx))
        else:
            ann_path = self.root / "val" / "val_annotations.txt"
            if not ann_path.exists():
                raise FileNotFoundError(f"Missing Tiny ImageNet file: {ann_path}")
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    filename, wnid = parts[0], parts[1]
                    if wnid not in self.class_to_idx:
                        continue
                    img_path = self.root / "val" / "images" / filename
                    if img_path.exists():
                        self.samples.append((str(img_path),
                                             self.class_to_idx[wnid]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for Tiny ImageNet split='{split}' in {self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, int(label)



DATASET_ALIASES: Dict[str, str] = {
    "caltech101":      "caltech101",
    "cifar10":         "cifar10",
    "cifar100":        "cifar100",
    "oxford_pets":     "oxford_pets",
    "oxfordpets":      "oxford_pets",
    "tiny_imagenet":   "tiny_imagenet",
    "tinyimagenet200": "tiny_imagenet",
    "tinyimagenet":    "tiny_imagenet",
}

DATASET_CHOICES = ["caltech101", "cifar10", "cifar100",
                   "oxford_pets", "tiny_imagenet"]

DATASET_NUM_CLASSES = {
    "caltech101": 101, "cifar10": 10, "cifar100": 100,
    "oxford_pets": 37, "tiny_imagenet": 200,
}


def canonicalize_dataset_name(name: str) -> str:
    key = name.lower()
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {DATASET_CHOICES} "
            f"(aliases: oxfordpets, tinyimagenet200)"
        )
    return DATASET_ALIASES[key]


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                 num_workers: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def build_loaders(dataset_name: str, data_root: str,
                  transform: A.Compose, batch_size: int,
                  num_workers: int,
                  seed: int = DEFAULT_SEED
                  ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_loader, test_loader, num_classes).

    """
    dataset_name = canonicalize_dataset_name(dataset_name)
    generator    = torch.Generator().manual_seed(seed)

    if dataset_name == "cifar10":
        tr_full = torchvision.datasets.CIFAR10(data_root, train=True,  download=True)
        te      = torchvision.datasets.CIFAR10(data_root, train=False, download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_full))
        train_n = len(tr_full) - val_n
        tr_sub, val_sub = random_split(tr_full, [train_n, val_n], generator=generator)
        return (_make_loader(AlbumentationsWrapper(tr_sub,  transform), batch_size, True,  num_workers),
                _make_loader(AlbumentationsWrapper(val_sub, transform), batch_size, False, num_workers),
                _make_loader(AlbumentationsWrapper(te,      transform), batch_size, False, num_workers),
                10)

    if dataset_name == "cifar100":
        tr_full = torchvision.datasets.CIFAR100(data_root, train=True,  download=True)
        te      = torchvision.datasets.CIFAR100(data_root, train=False, download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_full))
        train_n = len(tr_full) - val_n
        tr_sub, val_sub = random_split(tr_full, [train_n, val_n], generator=generator)
        return (_make_loader(AlbumentationsWrapper(tr_sub,  transform), batch_size, True,  num_workers),
                _make_loader(AlbumentationsWrapper(val_sub, transform), batch_size, False, num_workers),
                _make_loader(AlbumentationsWrapper(te,      transform), batch_size, False, num_workers),
                100)

    if dataset_name == "oxford_pets":
        tr_full = torchvision.datasets.OxfordIIITPet(data_root, split="trainval", download=True)
        te      = torchvision.datasets.OxfordIIITPet(data_root, split="test",     download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_full))
        train_n = len(tr_full) - val_n
        tr_sub, val_sub = random_split(tr_full, [train_n, val_n], generator=generator)
        return (_make_loader(AlbumentationsWrapper(tr_sub,  transform), batch_size, True,  num_workers),
                _make_loader(AlbumentationsWrapper(val_sub, transform), batch_size, False, num_workers),
                _make_loader(AlbumentationsWrapper(te,      transform), batch_size, False, num_workers),
                37)

    if dataset_name == "caltech101":
        raw     = torchvision.datasets.Caltech101(data_root, download=True)
        n       = len(raw)
        train_n = int(CALTECH_TRAIN_FRACTION * n)
        val_n   = int(CALTECH_VAL_FRACTION * n)
        test_n  = n - train_n - val_n         
        tr_sub, val_sub, te_sub = random_split(
            raw, [train_n, val_n, test_n], generator=generator)
        return (_make_loader(AlbumentationsWrapper(tr_sub,  transform), batch_size, True,  num_workers),
                _make_loader(AlbumentationsWrapper(val_sub, transform), batch_size, False, num_workers),
                _make_loader(AlbumentationsWrapper(te_sub,  transform), batch_size, False, num_workers),
                101)

    if dataset_name == "tiny_imagenet":
        tr_full = TinyImageNetDataset(data_root, split="train", transform=transform)
        te      = TinyImageNetDataset(data_root, split="val",   transform=transform)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_full))
        train_n = len(tr_full) - val_n
        tr_sub, val_sub = random_split(tr_full, [train_n, val_n], generator=generator)
        return (_make_loader(tr_sub,  batch_size, True,  num_workers),
                _make_loader(val_sub, batch_size, False, num_workers),
                _make_loader(te,      batch_size, False, num_workers),
                200)

    raise AssertionError("Unreachable")


class ProbeClassifier(nn.Module):
    """
    Frozen backbone (via adapter.extract_fn) + trainable MLP head.

    """
    def __init__(self, adapter: BackboneAdapter, in_dim: int,
                 num_classes: int, head: str, device: torch.device):
        super().__init__()
        self.adapter = adapter
        self.device  = device
        self.in_dim  = in_dim

        if head == "linear":
            self.head = nn.Linear(in_dim, num_classes)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(in_dim, HEAD_HIDDEN),
                nn.BatchNorm1d(HEAD_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Dropout(HEAD_DROPOUT),
                nn.Linear(HEAD_HIDDEN, num_classes),
            )
        else:
            raise ValueError(f"Unknown head '{head}'")

        # Lock backbone in eval
        self.adapter.module.eval()
        for p in self.adapter.module.parameters():
            p.requires_grad = False
        self.adapter.module.train = types.MethodType(
            lambda self_, mode=True: self_, self.adapter.module
        )

        self.head.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.adapter(images.to(self.device))   
        return self.head(feats)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.eval()
        correct_top1 = correct_top5 = total = 0
        for images, labels in tqdm(loader, desc="  eval", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self(images)
            total += labels.size(0)
            k = min(5, logits.size(1))
            _, topk = torch.topk(logits, k=k, dim=1)
            correct_top1 += (topk[:, 0] == labels).sum().item()
            correct_top5 += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()
        return {"top1": correct_top1 / max(total, 1),
                "top5": correct_top5 / max(total, 1)}


def train_probe(adapter: BackboneAdapter, in_dim: int,
                dataset_name: str, data_root: str,
                head: str, epochs: int, patience: int,
                batch_size: int, lr: float, weight_decay: float,
                num_workers: int, device: torch.device,
                out_dir: str, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    dataset_name = canonicalize_dataset_name(dataset_name)
    print(f"\n{'=' * 64}\n  Dataset : {dataset_name.upper()}\n{'=' * 64}")

    transform = get_transform(224, adapter.norm_mean, adapter.norm_std)
    train_loader, val_loader, test_loader, num_classes = build_loaders(
        dataset_name, data_root, transform, batch_size, num_workers, seed=seed)

    train_n = len(train_loader.dataset)
    val_n   = len(val_loader.dataset)
    test_n  = len(test_loader.dataset)
    print(f"  Split   : train={train_n}  val={val_n}  test={test_n}  classes={num_classes}")

    model = ProbeClassifier(adapter, in_dim, num_classes, head, device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=SCHED_ETA_MIN)
    criterion = nn.CrossEntropyLoss()

    best_val_top1, best_val_top5, best_epoch = -1.0, 0.0, 0
    bad_epochs, executed_epochs = 0, 0
    best_path = os.path.join(out_dir, f"best_{adapter.name}_{dataset_name}.pt")

    for epoch in range(epochs):
        executed_epochs = epoch + 1
        model.train()                     
        running_loss = correct = total = 0.0

        for images, labels in tqdm(train_loader,
                                   desc=f"  ep {epoch+1:3d}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        train_acc = correct / max(total, 1)
        val = model.evaluate(val_loader)

        print(f"  ep {epoch+1:3d} | loss {running_loss/max(len(train_loader),1):.4f}"
              f" | train_top1 {train_acc:.4f} | val_top1 {val['top1']:.4f}"
              f" | val_top5 {val['top5']:.4f} | lr {scheduler.get_last_lr()[0]:.1e}")

        if val["top1"] > best_val_top1:
            best_val_top1, best_val_top5, best_epoch = val["top1"], val["top5"], epoch + 1
            bad_epochs = 0
            torch.save(model.head.state_dict(), best_path)
            print(f"    -> new best val_top1={best_val_top1:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break


    model.head.load_state_dict(torch.load(best_path, map_location=device))
    test = model.evaluate(test_loader)
    print(f"\n  {dataset_name.upper()} best epoch {best_epoch}"
          f" | test_top1 {test['top1']:.4f} | test_top5 {test['top5']:.4f}")

    return {
        "dataset": dataset_name, "num_classes": num_classes,
        "train_size": train_n, "val_size": val_n, "test_size": test_n,
        "best_val_top1": round(best_val_top1, 4),
        "best_val_top5": round(best_val_top5, 4),
        "test_top1": round(test["top1"], 4),
        "test_top5": round(test["top5"], 4),
        "best_epoch": best_epoch, "total_epochs": executed_epochs,
    }



def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def print_results_table(results: List[Dict], head: str, backbone_name: str) -> None:
    print(f"\n\n{'=' * 72}")
    print(f"  Transfer Evaluation Summary  [{backbone_name}]")
    print(f"  Head : {head}")
    print(f"{'=' * 72}\n")
    print("| Dataset       | Classes | Val Top-1 | Val Top-5 | Test Top-1 | Test Top-5 | Best Epoch |")
    print("|---------------|---------|-----------|-----------|------------|------------|------------|")
    for r in results:
        print(f"| {str(r['dataset']):<13} | {str(r['num_classes']):<7} "
              f"| {_fmt(r['best_val_top1']):<9} | {_fmt(r['best_val_top5']):<9} "
              f"| {_fmt(r.get('test_top1', 'N/A')):<10} | {_fmt(r.get('test_top5', 'N/A')):<10} "
              f"| {str(r['best_epoch']):<10} |")
    finite_val = [r['best_val_top1'] for r in results
                  if isinstance(r['best_val_top1'], (int, float))]
    finite_test = [r['test_top1'] for r in results
                   if isinstance(r.get('test_top1'), (int, float))]
    if finite_val:
        print(f"\n  Mean Val Top-1  : {sum(finite_val)/len(finite_val):.4f}")
    if finite_test:
        print(f"  Mean Test Top-1 : {sum(finite_test)/len(finite_test):.4f}")
    print()


def save_results(results: List[Dict], out_dir: str,
                 backbone_name: str, model_path: str, head: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"backbone": backbone_name, "model_path": model_path,
            "head": head, "timestamp": ts, "results": results}
    with open(os.path.join(out_dir, f"probe_{backbone_name}_{head}_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, f"probe_{backbone_name}_{head}_{ts}.csv"), index=False)
    print(f"  Results saved to {out_dir}")


def run_probes(adapter: BackboneAdapter, datasets: List[str],
               data_root: str, out_dir: str, *,
               head: str = "mlp", epochs: int = 100, patience: int = 15,
               batch_size: int = 64, lr: float = 1e-3,
               weight_decay: float = 1e-4, num_workers: int = 4,
               device: torch.device = None, seed: int = DEFAULT_SEED,
               model_path: str = "") -> None:
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed, deterministic=True)

    if not datasets or datasets == ["all"]:
        datasets = DATASET_CHOICES
    else:
        datasets = [canonicalize_dataset_name(d) for d in datasets]

    in_dim = discover_feature_dim(adapter, device)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  Linear-probe transfer  |  backbone={adapter.name}")
    print(f"  feature dim (pre-proj, un-normalised) : {in_dim}")
    print(f"  head={head}  hidden={HEAD_HIDDEN}  lr={lr}  wd={weight_decay}")
    print(f"  norm mean={adapter.norm_mean}  std={adapter.norm_std}")
    print(f"  datasets={datasets}")
    print(f"{'=' * 64}")

    results: List[Dict] = []
    for ds in datasets:
        try:
            set_seed(seed, deterministic=True)   
            results.append(train_probe(
                adapter, in_dim, ds, data_root, head, epochs, patience,
                batch_size, lr, weight_decay, num_workers, device, out_dir, seed))
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {ds} failed:\n{traceback.format_exc()}")
            results.append({"dataset": ds, "num_classes": "_",
                            "train_size": "_", "val_size": "_", "test_size": "_",
                            "best_val_top1": "ERROR", "best_val_top5": "ERROR",
                            "test_top1": "ERROR", "test_top5": "ERROR",
                            "best_epoch": "_", "total_epochs": "_",
                            "error": str(e)})

    print_results_table(results, head, adapter.name)
    save_results(results, out_dir, adapter.name, model_path, head)