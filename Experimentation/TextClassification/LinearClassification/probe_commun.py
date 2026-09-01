import os
import json
import random
import types
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


SCHED_ETA_MIN = 1e-5
DEFAULT_SEED  = 42
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
class TextBackboneAdapter:
    module:      nn.Module
    extract_fn:  Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    tokenize_fn: Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]]
    name:        str = "backbone"

    def features(self, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        return self.extract_fn(input_ids, attention_mask)


def discover_feature_dim(adapter: TextBackboneAdapter,
                         device: torch.device) -> int:
    """One dummy forward through tokenize_fn + extract_fn to read width D."""
    adapter.module.eval()
    ids, mask = adapter.tokenize_fn(["a quick sanity check sentence",
                                     "another short example"])
    ids, mask = ids.to(device), mask.to(device)
    with torch.no_grad():
        feats = adapter.extract_fn(ids, mask)
    if feats.dim() != 2:
        raise RuntimeError(
            f"[{adapter.name}] extract_fn must return (B, D); got "
            f"{tuple(feats.shape)}. Pool inside the adapter.")
    return feats.shape[-1]



class TextDataset(Dataset):
    """Holds raw strings + int labels. Tokenisation happens in the collate."""
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts  = list(texts)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def make_collate(tokenize_fn: Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]]):
    """Collate that tokenises a batch of raw strings with the given tokenizer."""
    def _collate(batch):
        texts, labels = zip(*batch)
        ids, mask = tokenize_fn(list(texts))
        return ids, mask, torch.tensor(labels, dtype=torch.long)
    return _collate


DATASET_ALIASES: Dict[str, str] = {
    "emotion": "emotion", "sst2": "sst2", "sst-2": "sst2", "trec": "trec",
}
DATASET_CHOICES = ["emotion", "sst2", "trec"]


def canonicalize_dataset_name(name: str) -> str:
    key = name.lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {DATASET_CHOICES}")
    return DATASET_ALIASES[key]


def _loader(dataset: Dataset, batch_size: int, shuffle: bool,
            num_workers: int, collate) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def build_loaders(dataset_name: str, batch_size: int, num_workers: int,
                  collate, seed: int = DEFAULT_SEED
                  ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_loader, test_loader, num_classes).

    """
    from datasets import load_dataset as hf_load
    dataset_name = canonicalize_dataset_name(dataset_name)
    generator    = torch.Generator().manual_seed(seed)

    def ds(texts, labels): return TextDataset(list(texts), list(labels))

    if dataset_name == "emotion":
        d = hf_load("dair-ai/emotion")
        return (_loader(ds(d["train"]["text"],      d["train"]["label"]),      batch_size, True,  num_workers, collate),
                _loader(ds(d["validation"]["text"], d["validation"]["label"]), batch_size, False, num_workers, collate),
                _loader(ds(d["test"]["text"],       d["test"]["label"]),       batch_size, False, num_workers, collate),
                6)

    if dataset_name == "sst2":
        d = hf_load("stanfordnlp/sst2")
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
        d = hf_load("CogComp/trec", trust_remote_code=True)
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



class ProbeClassifier(nn.Module):
    def __init__(self, adapter: TextBackboneAdapter, in_dim: int,
                 num_classes: int, head: str, device: torch.device):
        super().__init__()
        self.adapter = adapter
        self.device  = device
        self.in_dim  = in_dim

        if head == "linear":
            self.head = nn.Linear(in_dim, num_classes)
        elif head == "mlp":
            hidden = 512
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(hidden, num_classes),
            )
        else:
            raise ValueError(f"Unknown head '{head}'")

        
        self.adapter.module.eval()
        for p in self.adapter.module.parameters():
            p.requires_grad = False
        self.adapter.module.train = types.MethodType(
            lambda self_, mode=True: self_, self.adapter.module)

        self.head.to(device)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.adapter.features(
                input_ids.to(self.device), attention_mask.to(self.device))
        return self.head(feats)

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


def train_probe(adapter: TextBackboneAdapter, in_dim: int,
                dataset_name: str, head: str, epochs: int, patience: int,
                batch_size: int, lr: float, weight_decay: float,
                num_workers: int, device: torch.device,
                out_dir: str, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    dataset_name = canonicalize_dataset_name(dataset_name)
    print(f"\n{'=' * 64}\n  Dataset : {dataset_name.upper()}\n{'=' * 64}")

    collate = make_collate(adapter.tokenize_fn)
    train_loader, val_loader, test_loader, num_classes = build_loaders(
        dataset_name, batch_size, num_workers, collate, seed=seed)

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

    best_val, best_epoch, bad, executed = -1.0, 0, 0, 0
    best_path = os.path.join(out_dir, f"best_{adapter.name}_{dataset_name}.pt")

    for epoch in range(epochs):
        executed = epoch + 1
        model.train()
        running = correct = total = 0.0
        for ids, mask, labels in tqdm(train_loader,
                                      desc=f"  ep {epoch+1:3d}/{epochs}", leave=False):
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        scheduler.step()

        train_acc = correct / max(total, 1)
        val = model.evaluate(val_loader)
        print(f"  ep {epoch+1:3d} | loss {running/max(len(train_loader),1):.4f}"
              f" | train_acc {train_acc:.4f} | val_acc {val['top1']:.4f}"
              f" | lr {scheduler.get_last_lr()[0]:.1e}")

        if val["top1"] > best_val:
            best_val, best_epoch, bad = val["top1"], epoch + 1, 0
            torch.save(model.head.state_dict(), best_path)
            print(f"    -> new best val_acc={best_val:.4f}")
        else:
            bad += 1
            if bad >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

   
    model.head.load_state_dict(torch.load(best_path, map_location=device))
    test_top1 = round(model.evaluate(test_loader)["top1"], 4)
    print(f"\n  {dataset_name.upper()} best epoch {best_epoch} | test_acc {test_top1:.4f}")

    return {
        "dataset": dataset_name, "num_classes": num_classes,
        "train_size": train_n, "val_size": val_n, "test_size": test_n,
        "best_val_top1": round(best_val, 4),
        "test_top1": test_top1,
        "best_epoch": best_epoch, "total_epochs": executed,
    }


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def print_results_table(results: List[Dict], head: str, backbone_name: str) -> None:
    print(f"\n\n{'=' * 72}")
    print(f"  Text Transfer Evaluation Summary  [{backbone_name}]")
    print(f"  Head : {head}")
    print(f"{'=' * 72}\n")
    print("| Dataset  | Classes | Val Acc   | Test Acc  | Best Epoch |")
    print("|----------|---------|-----------|-----------|------------|")
    for r in results:
        print(f"| {str(r['dataset']):<8} | {str(r['num_classes']):<7} "
              f"| {_fmt(r['best_val_top1']):<9} | {_fmt(r.get('test_top1','N/A')):<9} "
              f"| {str(r['best_epoch']):<10} |")
    finite_val = [r['best_val_top1'] for r in results
                  if isinstance(r['best_val_top1'], (int, float))]
    finite_test = [r['test_top1'] for r in results
                   if isinstance(r.get('test_top1'), (int, float))]
    if finite_val:
        print(f"\n  Mean Val Acc  : {sum(finite_val)/len(finite_val):.4f}")
    if finite_test:
        print(f"  Mean Test Acc : {sum(finite_test)/len(finite_test):.4f}")
    print()


def save_results(results: List[Dict], out_dir: str,
                 backbone_name: str, model_path: str, head: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {"backbone": backbone_name, "model_path": model_path,
            "head": head, "timestamp": ts, "results": results}
    with open(os.path.join(out_dir, f"text_probe_{backbone_name}_{head}_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, f"text_probe_{backbone_name}_{head}_{ts}.csv"), index=False)
    print(f"  Results saved to {out_dir}")



def run_probes(adapter: TextBackboneAdapter, datasets: List[str], out_dir: str, *,
               head: str = "linear", epochs: int = 100, patience: int = 15,
               batch_size: int = 64, lr: float = 1e-4, weight_decay: float = 1e-4,
               num_workers: int = 4, device: torch.device = None,
               seed: int = DEFAULT_SEED, model_path: str = "") -> None:
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(seed, deterministic=True)

    datasets = (DATASET_CHOICES if not datasets or datasets == ["all"]
                else [canonicalize_dataset_name(d) for d in datasets])

    in_dim = discover_feature_dim(adapter, device)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  Text linear-probe transfer  |  backbone={adapter.name}")
    print(f"  feature dim (pre-proj, un-normalised) : {in_dim}")
    print(f"  head={head}  lr={lr}  wd={weight_decay}")
    print(f"  datasets={datasets}")
    print(f"{'=' * 64}")

    results: List[Dict] = []
    for ds in datasets:
        try:
            set_seed(seed, deterministic=True)
            results.append(train_probe(
                adapter, in_dim, ds, head, epochs, patience,
                batch_size, lr, weight_decay, num_workers, device, out_dir, seed))
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {ds} failed:\n{traceback.format_exc()}")
            results.append({"dataset": ds, "num_classes": "_", "train_size": "_",
                            "val_size": "_", "test_size": "_",
                            "best_val_top1": "ERROR", "test_top1": "ERROR",
                            "best_epoch": "_", "total_epochs": "_", "error": str(e)})

    print_results_table(results, head, adapter.name)
    save_results(results, out_dir, adapter.name, model_path, head)