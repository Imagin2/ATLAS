import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import warnings
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
from einops.layers.torch import Rearrange
from dataclasses import dataclass

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataset.dataset import PatchExtractor
from src.utils.metrics import set_seed  
from src.models.ModalityAwareBasisEmbedder import  LearnablePositionalEncoding
from src.models.model import AttentionPooling, UniversalProjectionEncoder


DEFAULT_SEED = 42
VAL_FRACTION_FROM_TRAIN = 0.1     # CIFAR-10/100, Oxford-Pets (90/10 train/val)
# Caltech101 
CALTECH_TRAIN_FRACTION = 0.8
CALTECH_VAL_FRACTION   = 0.1
CALTECH_TEST_FRACTION  = 0.1      
# Tiny-ImageNet ships train + 'val' (no labeled test) 



class ImageStemEmbedder(nn.Module):
    """
    Image projection + positional-encoding stem.

    """

    def __init__(self, patch_dim: int, d_model: int, d_inter: int,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.img_proj_1 = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_inter),
        )
       
        self.shared_proj_2 = nn.Sequential(
            nn.Linear(d_inter, d_model),
            nn.LayerNorm(d_model),
        )
        self.content_pos_enc = LearnablePositionalEncoding(d_model, max_len=max_seq_len)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
    
        x = self.img_proj_1(patches)      
        x = self.shared_proj_2(x)         
        x = self.content_pos_enc(x)       
        return x


@dataclass
class UniversalProjectionOutput:
    last_hidden_state: torch.Tensor
    mean_output:       torch.Tensor
    pooler_output:     torch.Tensor
    cls_output:        torch.Tensor
    attn_pool_output:  torch.Tensor
    attn_weights:      Optional[torch.Tensor]


@dataclass
class ModalityEncodingOutput:
    feature:          torch.Tensor
    embedding:        torch.Tensor
    sequence:         torch.Tensor
    cls_output:       torch.Tensor
    attn_pool_output: torch.Tensor
    mean_output:      torch.Tensor
    attn_weights:     Optional[torch.Tensor]




class UniversalProjectionEncoder(nn.Module):
    """
    Shared Pre-LN transformer encoder for image and text.

    """
    def __init__(self, d_model: int = 512, num_head: int = 8,
                 num_layers: int = 4, dropout_rate: float = 0.1,
                 pooling_type: str = "attn"):
        super().__init__()
        self.pooling_type = pooling_type.lower()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_head,
            dim_feedforward=d_model * 4, dropout=dropout_rate,
            activation="gelu", batch_first=True,
            norm_first=True,             
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,  
        )
       
        self.output_norm    = nn.LayerNorm(d_model)
        self.cls_image      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_text       = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention_pool = AttentionPooling(d_model=d_model, dropout=dropout_rate)

    def forward(self, EX: torch.Tensor, modality: str,
                attention_mask: Optional[torch.Tensor] = None
                ) -> UniversalProjectionOutput:
        B      = EX.size(0)
        device = EX.device
        modality = modality.lower()

        cls_token = (self.cls_image if modality == "image" else self.cls_text
                     ).expand(B, 1, -1)
        x = torch.cat([cls_token, EX], dim=1)          

        if attention_mask is not None:
            cls_mask     = torch.ones(B, 1, dtype=attention_mask.dtype, device=device)
            full_mask    = torch.cat([cls_mask, attention_mask], dim=1)
            padding_mask = ~full_mask.bool()
        else:
            padding_mask = None

       
        encoded = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        encoded = self.output_norm(encoded)             

        cls_output     = encoded[:, 0, :]
        content_tokens = encoded[:, 1:, :]

        if attention_mask is not None:
            mask_f      = attention_mask.bool().unsqueeze(-1).float()
            mean_output = (content_tokens * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6)
        else:
            mean_output = content_tokens.mean(dim=1)

        attn_pool_output, attn_weights = self.attention_pool(
            content_tokens, attention_mask=attention_mask
        )

        if self.pooling_type == "cls":
            pooler_output = cls_output
        elif self.pooling_type == "attn":
            pooler_output = attn_pool_output
        elif self.pooling_type == "mean":
            pooler_output = mean_output
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type!r}")

        return UniversalProjectionOutput(
            last_hidden_state=encoded,
            mean_output=mean_output,
            pooler_output=pooler_output,
            cls_output=cls_output,
            attn_pool_output=attn_pool_output,
            attn_weights=attn_weights,
        )



class UnimodalImageModel(nn.Module):
    """
    End-to-end image classifier.
    
    The model is trained end-to-end from scratch on image labels only.
    """

    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 channels: int,
                 d_model: int,
                 d_inter: int,
                 num_head: int,
                 num_layers: int,
                 dropout: float,
                 max_seq_len: int,
                 num_classes: int,
                 head: str = "mlp",
                 pooling: str = "attn"):
        super().__init__()

        #  Stage 1: patch extraction  
        self.patch_extractor = PatchExtractor(image_size, patch_size, channels)
        patch_dim = self.patch_extractor.patch_dim

        #  Stage 2: projection + positional encoding 
        self.stem_embedder = ImageStemEmbedder(
            patch_dim=patch_dim,
            d_model=d_model,
            d_inter=d_inter,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        #  Stage 3:Universal Encoder 
        self.universal_encoder = UniversalProjectionEncoder(
            d_model=d_model, num_head=num_head,
            num_layers=num_layers, dropout_rate=dropout,
            pooling_type=pooling,
        )
       

        #  Stage 4: classification head 
        
        if head == "linear":
            self.classifier = nn.Linear(d_model, num_classes)
        elif head == "mlp":
            hidden = max(512, d_model)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden // 2, num_classes),
            )
        else:
            raise ValueError(f"Unknown head '{head}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        
        patches = self.patch_extractor(x)           
        tokens  = self.stem_embedder(patches)      

        # Stage 3: Transformer + output norm
        
        encoded = self.universal_encoder(tokens, modality="image", attention_mask=None)  
        feat = encoded.pooler_output  
        return self.classifier(feat)

                   

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device) -> Dict[str, float]:
        self.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for inputs, labels in tqdm(loader, desc="  eval", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = self(inputs)
            total += labels.size(0)

            k = min(5, logits.size(1))
            _, topk = torch.topk(logits, k=k, dim=1)
            correct_top1 += (topk[:, 0] == labels).sum().item()
            correct_top5 += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()

        return {
            "top1": correct_top1 / max(total, 1),
            "top5": correct_top5 / max(total, 1),
        }


def _make_randaugment_equiv() -> list:
    """Version-aware RandAugment substitute."""
    import albumentations as _A
    if hasattr(_A, "RandAugment"):
        return [_A.RandAugment(num_ops=2, magnitude=9, p=0.7)]
   
    pool = [
        _A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
        _A.Equalize(p=1.0),
        _A.Sharpen(alpha=(0.1, 0.4), lightness=(0.7, 1.3), p=1.0),
        _A.Posterize(num_bits=(4, 8), p=1.0),
        
        _A.Solarize(p=1.0),
        _A.Rotate(limit=30, p=1.0),
        _A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1.0),
        _A.AutoContrast(p=1.0) if hasattr(_A, "AutoContrast") else _A.CLAHE(p=1.0),
    ]
    if hasattr(_A, "SomeOf"):
        return [_A.SomeOf(pool, n=2, p=0.7)]
    return [
        _A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.4),
        _A.Equalize(p=0.2),
        _A.Sharpen(p=0.2),
        _A.Rotate(limit=20, p=0.3),
    ]


def _make_coarse_dropout() -> object:
    """
    Version-aware CoarseDropout 
    """
    import albumentations as _A
    try:
        
        return _A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.25,
        )
    except TypeError:
        
        return _A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            p=0.25,
        )
def get_train_transform(image_size: int) -> A.Compose:
    return A.Compose([
        # RandomResizedCrop instead of Resize 
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        *_make_randaugment_equiv(),  
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.ToGray(p=0.1),
        _make_coarse_dropout(),       
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_eval_transform(image_size: int) -> A.Compose:
    """Identical to eval_transfer.py get_eval_transform."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
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
    """
    Loads Tiny ImageNet from <root>/tiny-imagenet-200/.
    """

    def __init__(self, root: str, split: str = "train", transform: A.Compose = None):
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
                        self.samples.append((str(img_path), self.class_to_idx[wnid]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for Tiny ImageNet split='{split}' in {self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, int(label)



DATASET_ALIASES: Dict[str, str] = {
    "caltech101":     "caltech101",
    "cifar10":        "cifar10",
    "cifar100":       "cifar100",
    "oxford_pets":    "oxford_pets",
    "oxfordpets":     "oxford_pets",
    "tiny_imagenet":  "tiny_imagenet",
    "tinyimagenet200":"tiny_imagenet",
}

DATASET_CHOICES = ["caltech101", "cifar10", "cifar100", "oxford_pets", "tiny_imagenet"]


def canonicalize_dataset_name(name: str) -> str:
    if name not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {DATASET_CHOICES} "
            f"(aliases: oxfordpets, tinyimagenet200)"
        )
    return DATASET_ALIASES[name]


def make_loader(dataset: Dataset, batch_size: int,
                shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def build_loaders(dataset_name: str,
                  data_root: str,
                  transform_train: A.Compose,
                  transform_eval: A.Compose,
                  batch_size: int,
                  num_workers: int,
                  seed: int = DEFAULT_SEED
                  ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_loader, test_loader, num_classes).

    """
    dataset_name = canonicalize_dataset_name(dataset_name)
    generator = torch.Generator().manual_seed(seed)

    if dataset_name == "cifar10":
        tr_raw = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True)
        te_raw = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_raw))
        train_n = len(tr_raw) - val_n
        tr_sub, val_sub = random_split(tr_raw, [train_n, val_n], generator=generator)
        return (make_loader(AlbumentationsWrapper(tr_sub,  transform_train), batch_size, True,  num_workers),
                make_loader(AlbumentationsWrapper(val_sub, transform_eval),  batch_size, False, num_workers),
                make_loader(AlbumentationsWrapper(te_raw,  transform_eval),  batch_size, False, num_workers),
                10)

    if dataset_name == "cifar100":
        tr_raw = torchvision.datasets.CIFAR100(root=data_root, train=True,  download=True)
        te_raw = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_raw))
        train_n = len(tr_raw) - val_n
        tr_sub, val_sub = random_split(tr_raw, [train_n, val_n], generator=generator)
        return (make_loader(AlbumentationsWrapper(tr_sub,  transform_train), batch_size, True,  num_workers),
                make_loader(AlbumentationsWrapper(val_sub, transform_eval),  batch_size, False, num_workers),
                make_loader(AlbumentationsWrapper(te_raw,  transform_eval),  batch_size, False, num_workers),
                100)

    if dataset_name == "oxford_pets":
        tr_raw = torchvision.datasets.OxfordIIITPet(root=data_root, split="trainval", download=True)
        te_raw = torchvision.datasets.OxfordIIITPet(root=data_root, split="test",     download=True)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_raw))
        train_n = len(tr_raw) - val_n
        tr_sub, val_sub = random_split(tr_raw, [train_n, val_n], generator=generator)
        return (make_loader(AlbumentationsWrapper(tr_sub,  transform_train), batch_size, True,  num_workers),
                make_loader(AlbumentationsWrapper(val_sub, transform_eval),  batch_size, False, num_workers),
                make_loader(AlbumentationsWrapper(te_raw,  transform_eval),  batch_size, False, num_workers),
                37)

    if dataset_name == "caltech101":
        raw     = torchvision.datasets.Caltech101(root=data_root, download=True)
        n       = len(raw)
        train_n = int(CALTECH_TRAIN_FRACTION * n)
        val_n   = int(CALTECH_VAL_FRACTION * n)
        test_n  = n - train_n - val_n         
        tr_sub, val_sub, te_sub = random_split(raw, [train_n, val_n, test_n], generator=generator)
        return (make_loader(AlbumentationsWrapper(tr_sub,  transform_train), batch_size, True,  num_workers),
                make_loader(AlbumentationsWrapper(val_sub, transform_eval),  batch_size, False, num_workers),
                make_loader(AlbumentationsWrapper(te_sub,  transform_eval),  batch_size, False, num_workers),
                101)

    if dataset_name == "tiny_imagenet":
        tr_full = TinyImageNetDataset(data_root, split="train", transform=transform_train)
        te      = TinyImageNetDataset(data_root, split="val",   transform=transform_eval)
        val_n   = int(VAL_FRACTION_FROM_TRAIN * len(tr_full))
        train_n = len(tr_full) - val_n
        tr_sub, val_sub = random_split(tr_full, [train_n, val_n], generator=generator)
        return (make_loader(tr_sub,  batch_size, True,  num_workers),
                make_loader(val_sub, batch_size, False, num_workers),
                make_loader(te,      batch_size, False, num_workers),
                200)

    raise AssertionError("Unreachable")



def train_one_dataset(dataset_name: str,
                      args: argparse.Namespace,
                      device: torch.device,
                      out_dir: str) -> Dict[str, object]:
    """
    Trains UnimodalImage end-to-end on one dataset.
    
    """
    dataset_name = canonicalize_dataset_name(dataset_name)

    print(f"\n{'=' * 64}")
    print(f"  Dataset : {dataset_name.upper()}  [UnimodalImage]")
    print(f"{'=' * 64}")

    transform_train = get_train_transform(args.image_size)
    transform_eval  = get_eval_transform(args.image_size)

    train_loader, val_loader, test_loader, num_classes = build_loaders(
        dataset_name=dataset_name,
        data_root=args.data_root,
        transform_train=transform_train,
        transform_eval=transform_eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    train_n = len(train_loader.dataset)
    val_n   = len(val_loader.dataset)
    test_n  = len(test_loader.dataset)
    print(f"  Split   : train={train_n}  val={val_n}  test={test_n}  classes={num_classes}")

    model = UnimodalImageModel(
        image_size=args.image_size,
        patch_size=args.patch_size,
        channels=args.channels,
        d_model=args.d_model,
        d_inter=args.d_inter,
        num_head=args.num_head,
        num_layers=args.num_layers,
        dropout=args.dropout_rate,
        max_seq_len=args.max_seq_len,
        num_classes=num_classes,
        head=args.head,
        pooling=args.pooling_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params  : {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    
    num_training_steps = args.epochs * len(train_loader)
    warmup_steps       = int(args.warmup_ratio * num_training_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = float(step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler    = GradScaler(enabled=args.use_amp)

    best_val_top1  = -1.0
    best_val_top5  = 0.0
    best_epoch     = 0
    bad_epochs     = 0
    executed_epochs = 0
    best_path      = os.path.join(out_dir, f"best_{dataset_name}.pt")

    for epoch in range(args.epochs):
        executed_epochs = epoch + 1
        model.train()

        running_loss = 0.0
        correct      = 0
        total        = 0

        for inputs, labels in tqdm(train_loader,
                                   desc=f"  ep {epoch + 1:3d}/{args.epochs}",
                                   leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.use_amp):
                logits = model(inputs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

        train_acc = correct / max(total, 1)
        val_acc   = model.evaluate(val_loader, device)
        val_top1  = val_acc["top1"]

        print(
            f"  ep {epoch + 1:3d} | "
            f"loss {running_loss / max(len(train_loader), 1):.4f} | "
            f"train_top1 {train_acc:.4f} | "
            f"val_top1 {val_top1:.4f} | "
            f"val_top5 {val_acc['top5']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top5 = val_acc["top5"]
            best_epoch    = epoch + 1
            bad_epochs    = 0
            torch.save(model.state_dict(), best_path)
            print(f"    -> new best val_top1={best_val_top1:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  Early stopping at epoch {epoch + 1} (patience={args.patience})")
                break


    model.load_state_dict(torch.load(best_path, map_location=device))
    reloaded_val = model.evaluate(val_loader,  device)
    test_acc     = model.evaluate(test_loader, device)
    print(f"\n  {dataset_name.upper()} best epoch {best_epoch}"
          f" | test_top1 {test_acc['top1']:.4f} | test_top5 {test_acc['top5']:.4f}")

    return {
        "dataset":          dataset_name,
        "num_classes":      num_classes,
        "train_size":       train_n,
        "val_size":         val_n,
        "test_size":        test_n,
        "best_val_top1":    round(best_val_top1, 4),
        "best_val_top5":    round(best_val_top5, 4),
        "reloaded_val_top1": round(reloaded_val["top1"], 4),
        "test_top1":        round(test_acc["top1"], 4),
        "test_top5":        round(test_acc["top5"], 4),
        "best_epoch":       best_epoch,
        "total_epochs":     executed_epochs,
        "n_params":         n_params,
    }



def _fmt(x) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def print_results_table(results: List[Dict[str, object]], args: argparse.Namespace) -> None:
    print(f"\n\n{'=' * 72}")
    print("  UnimodalImage Transfer Evaluation Summary")
    print(f"  Head     : {args.head}")
    print(f"  Pooling  : {args.pooling_type}")
    print(f"  d_model  : {args.d_model}  d_inter: {args.d_inter}  layers: {args.num_layers}  heads: {args.num_head}")
    print(f"{'=' * 72}\n")
    print("| Dataset       | Classes | Val Top-1   | Val Top-5   | Test Top-1  | Test Top-5  | Best Epoch |")
    print("|---------------|---------|-------------|-------------|-------------|-------------|------------|")
    for r in results:
        print(
            f"| {str(r['dataset']):<13} "
            f"| {str(r['num_classes']):<7} "
            f"| {_fmt(r['best_val_top1']):<11} "
            f"| {_fmt(r['best_val_top5']):<11} "
            f"| {_fmt(r.get('test_top1', 'N/A')):<11} "
            f"| {_fmt(r.get('test_top5', 'N/A')):<11} "
            f"| {str(r['best_epoch']):<10}|"
        )
    finite_val  = [r['best_val_top1'] for r in results
                   if isinstance(r['best_val_top1'], (int, float))]
    finite_test = [r['test_top1'] for r in results
                   if isinstance(r.get('test_top1'), (int, float))]
    if finite_val:
        print(f"\n  Mean Val Top-1  : {sum(finite_val)/len(finite_val):.4f}")
    if finite_test:
        print(f"  Mean Test Top-1 : {sum(finite_test)/len(finite_test):.4f}")
    print()


def save_results(results: List[Dict[str, object]], out_dir: str,
                 args: argparse.Namespace) -> None:
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta = {
        "model":     "UnimodalImage",
        "head":      args.head,
        "pooling":   args.pooling_type,
        "d_model":   args.d_model,
        "num_layers": args.num_layers,
        "timestamp": timestamp,
        "results":   results,
    }

    json_path = os.path.join(out_dir, f"unimodal_image_results_{timestamp}.json")
    csv_path  = os.path.join(out_dir, f"unimodal_image_results_{timestamp}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame(results).to_csv(csv_path, index=False)

    print("  Results saved:")
    print(f"    JSON : {json_path}")
    print(f"    CSV  : {csv_path}")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    datasets = args.datasets
    if datasets is None or datasets == ["all"]:
        datasets = DATASET_CHOICES
    else:
        datasets = [canonicalize_dataset_name(ds) for ds in datasets]

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print("  UnimodalImage End-to-End Image Classification Baseline")
    print(f"  Head     : {args.head}")
    print(f"  Pooling  : {args.pooling_type}")
    print(f"  Datasets : {datasets}")
    print(f"  Device   : {device}")
    print(f"  Out dir  : {out_dir}")
    print(f"{'=' * 64}")

    all_results: List[Dict[str, object]] = []
    for ds_name in datasets:
        try:
            result = train_one_dataset(
                dataset_name=ds_name,
                args=args,
                device=device,
                out_dir=out_dir,
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {ds_name} failed:")
            print(traceback.format_exc())
            all_results.append({
                "dataset":       ds_name,
                "num_classes":   "_",
                "train_size":    "_",
                "val_size":      "_",
                "test_size":     "_",
                "best_val_top1": "ERROR",
                "best_val_top5": "ERROR",
                "test_top1":     "ERROR",
                "test_top5":     "ERROR",
                "best_epoch":    "_",
                "total_epochs":  "_",
                "error":         str(e),
            })

    print_results_table(all_results, args)
    save_results(all_results, out_dir, args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UnimodalImage end-to-end supervised image classification baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    

    # Evaluation
    ev = p.add_argument_group("evaluation")
    ev.add_argument(
        "--datasets", nargs="+", default=["all"],
        choices=DATASET_CHOICES + ["all", "oxfordpets", "tinyimagenet200"],
        metavar="DATASET",
        help=f"Datasets to evaluate. Choices: {DATASET_CHOICES}",
    )
    ev.add_argument("--head", type=str, default="mlp", choices=["linear", "mlp"],
                    help="Classification head: 'linear' or 'mlp'")
    ev.add_argument("--epochs",       type=int,   default=100)
    ev.add_argument("--patience",     type=int,   default=30)
    ev.add_argument("--batch_size",   type=int,   default=64)
    ev.add_argument("--lr",           type=float, default=1e-3)
    ev.add_argument("--weight_decay", type=float, default=0.0001)
    ev.add_argument("--max_grad_norm", type=float, default=1.0)
    ev.add_argument("--label_smoothing", type=float, default=0.1,
                    help="Label smoothing")
    ev.add_argument("--warmup_ratio", type=float, default=0.06,
                    help="Fraction of total steps used for linear LR warmup "
                    )
    ev.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.set_defaults(use_amp=True)
    ev.add_argument("--num_workers",  type=int,   default=4)
    ev.add_argument("--seed",         type=int,   default=42)
    ev.add_argument("--device",       type=str,   default=None)

    # Paths
    pa = p.add_argument_group("paths")
    pa.add_argument("--data_root",  type=str,
                    default="./data")
    pa.add_argument("--output_dir", type=str,
                    default="./Results/UnimodalImage",
                    help="Directory for checkpoints and result files")


    ar = p.add_argument_group("architecture")
    ar.add_argument("--image_size",   type=int,   default=224)
    ar.add_argument("--patch_size",   type=int,   default=16)
    ar.add_argument("--channels",     type=int,   default=3)
    ar.add_argument("--d_model",      type=int,   default=512)
    ar.add_argument("--d_inter",      type=int,   default=512,
                    help="Intermediate projection dim ")
    ar.add_argument("--num_head",     type=int,   default=4)
    ar.add_argument("--num_layers",   type=int,   default=4)
    ar.add_argument("--dropout_rate", type=float, default=0.1)
    ar.add_argument("--max_seq_len",  type=int,   default=512,
                    help="Max sequence length for LearnablePositionalEncoding")
    ar.add_argument("--pooling_type", type=str,   default="attn",
                    choices=["attn", "mean", "cls"])

    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()
    main(args)