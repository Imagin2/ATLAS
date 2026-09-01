
import math
import torch
from src.utils.metrics import MetricTracker, METRIC_KEYS
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import torch.nn as nn

class AvgMeter:
    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg = self.sum = self.count = 0.0

    def update(self, val: float, count: int = 1):
        self.count += count
        self.sum   += val * count
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer) -> float:
    for pg in optimizer.param_groups:
        return pg["lr"]


def get_cosine_schedule(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    lr_min: float,
    lr_max: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min / lr_max + (1.0 - lr_min / lr_max) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, train_loader, optimizer,
                    scaler: GradScaler, use_amp: bool = True,
                    
                    max_grad_norm: float = 1.0,
                    epoch: int = 0,
                    is_main: bool = True):
    model.train()
    loss_meter     = AvgMeter("train_loss")
    metric_tracker = MetricTracker(METRIC_KEYS)
   
    tqdm_object = tqdm(train_loader, total=len(train_loader), disable=not is_main)
    nan_batches = 0

    for batch_idx, batch in enumerate(tqdm_object):
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            loss, metrics = model(batch, return_metrics=True,
                                  )

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            nan_batches += 1
            if is_main:
                tqdm_object.set_postfix(
                    WARNING=f"NaN/Inf loss skipped ({nan_batches} so far)",
                    batch=batch_idx,
                )
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        count = batch["image"].size(0)
        loss_meter.update(loss_val, count)
        metric_tracker.update(metrics, count)

        if is_main:
            tqdm_object.set_postfix(
                loss     = f"{loss_meter.avg:.4f}",
                lr       = f"{get_lr(optimizer):.2e}",
            )

    if nan_batches > 0 and is_main:
        print(f"  [NaN guard] {nan_batches} batch(es) skipped this epoch.")
    return loss_meter, metric_tracker


def valid_one_epoch(model, valid_loader,
                    use_amp: bool = True, 
                    is_main: bool = True):
    model.eval()
    loss_meter     = AvgMeter("val_loss")
    metric_tracker = MetricTracker(METRIC_KEYS)
    tqdm_object = tqdm(valid_loader, total=len(valid_loader), disable=not is_main)

    with torch.no_grad():
        for batch in tqdm_object:
            with autocast(enabled=use_amp):
                loss, metrics = model(batch, return_metrics=True,
                                      )
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
            metric_tracker.update(metrics, count)
            if is_main:
                tqdm_object.set_postfix(
                    loss    = f"{loss_meter.avg:.4f}",

                    r1      = f"{metrics.get('recall@1', 0.0):.4f}",

                )
    return loss_meter, metric_tracker


