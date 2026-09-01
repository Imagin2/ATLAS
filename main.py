import datetime
import math
import os
import random
import torch
from tqdm import tqdm
from torch.cuda.amp import  GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.dataset.dataset import *
from src.dataset.tokenizer import build_tokenizer,build_corpus,train_spm
from src.utils.metrics import *
from src.models.train_valid import *
from src.models.model import build_stage1_model
from config import parse_args

def main():
  
    is_ddp     = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if is_ddp:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=3),
        )
        torch.cuda.set_device(local_rank)

    args    = parse_args()
    cfg     = CFG(**vars(args))
    is_main = cfg.is_main

    rank_seed = cfg.seed if is_main else cfg.seed + local_rank
    set_seed(rank_seed, deterministic=getattr(cfg, "deterministic", False))

    
    coco_tr, flickr_tr, tcaps_tr = make_pairs(
        cfg.train_annotation_file, cfg.image_dir,
        cfg=cfg, max_captions=cfg.max_captions, train=True,
    )
    coco_val, _, tcaps_val = make_pairs(
        cfg.val_annotation_file, cfg.image_dir_val,
        cfg=cfg, max_captions=cfg.max_captions, train=False,
    )
    all_pairs = coco_tr + flickr_tr + tcaps_tr + coco_val + tcaps_val
   
    random.shuffle(all_pairs)

   
    if cfg.tokenizer == "sentencepiece":
        spm_prefix  = cfg.spm_model_path.replace(".model", "")
        corpus_path = spm_prefix + "_corpus.txt"
        if not os.path.exists(cfg.spm_model_path):
            if is_main:
                print("[SPM] Training tokenizer from corpus ...")
                build_corpus(all_pairs, corpus_path)
                train_spm(corpus_path, spm_prefix, vocab_size=cfg.vocab_size)
            if is_ddp:
                dist.barrier()   
        else:
            if is_main:
                print(f"[SPM] Loading from {cfg.spm_model_path}")

    tokenizer = build_tokenizer(cfg)
    if is_main:
        print(f"  [Tokenizer] vocab_size={tokenizer.vocab_size}  "
              f"(cfg.vocab_size={cfg.vocab_size})")

   
    training_pairs   = all_pairs[:-cfg.val_size]
    validation_pairs = all_pairs[-cfg.val_size:]

    if getattr(cfg, "fast_dev_run", False):
        if is_main:
            print("\n  [fast_dev_run] capping data and epochs.")
        training_pairs   = training_pairs[:cfg.batch_size * 50]
        validation_pairs = validation_pairs[:cfg.batch_size * 10]
        cfg.epochs       = 2

    if is_main:
        print(f"\nTraining pairs  : {len(training_pairs)}")
        print(f"Validation pairs: {len(validation_pairs)}")

   
   
    train_loader = build_loader(training_pairs,   tokenizer, cfg=cfg, mode="train")
    val_loader   = build_loader(validation_pairs, tokenizer, cfg=cfg, mode="valid")

   
    model     = build_stage1_model(cfg)
    raw_model = model           

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        if is_main:
            print(f"  [DDP] Wrapped with DistributedDataParallel, world_size={dist.get_world_size()}")

    scaler    = GradScaler(enabled=cfg.use_amp)
    optimizer = torch.optim.AdamW(raw_model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = get_cosine_schedule(
        optimizer,
        warmup_epochs = cfg.warmup_epochs,
        total_epochs  = cfg.epochs,
        lr_min        = cfg.lr_min,
        lr_max        = cfg.lr,
    )

    results_logger = ResultsLogger(cfg.metrics_dir, run_name=cfg.run_name) if is_main else None
    start_epoch    = 0
    best_loss      = float("inf")
    num_bad_epochs = 0
    best_epoch     = 0

    
    if cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        if is_main:
            print(f"\n[Resume] Loading: {cfg.resume_checkpoint}")
        start_epoch, best_loss, num_bad_epochs, best_epoch = load_checkpoint(
            cfg.resume_checkpoint, raw_model, optimizer, lr_scheduler,
            cfg.device, cfg=cfg)
    else:
        if is_main:
            print("\n[Resume] Starting from scratch.")

    
    for epoch in range(start_epoch, cfg.epochs):
        if is_main:
            print(f"\n{'='*72}")
            print(f"  Epoch {epoch+1}/{cfg.epochs}  |  lr={get_lr(optimizer):.2e}"
                  f"  |  seed={cfg.seed}")
            print(f"{'='*72}")

        train_loss, train_metrics = train_one_epoch(
            model=model, train_loader=train_loader, optimizer=optimizer,
            scaler=scaler, use_amp=cfg.use_amp,
           
            max_grad_norm=cfg.max_grad_norm,
            epoch=epoch,
            is_main=is_main,
        )
        val_loss, val_metrics = valid_one_epoch(
            model=model, valid_loader=val_loader,
            use_amp=cfg.use_amp, 
            is_main=is_main,
        )

        
        if not is_main:
            continue

        t_avgs = train_metrics.avgs
        v_avgs = val_metrics.avgs

        epoch_is_nan = not math.isfinite(val_loss.avg)
        if epoch_is_nan:
            print(f"\n  [NaN Epoch] val_loss={val_loss.avg}  "
                  f"skipping best.pt + scheduler.")
            if epoch > 0:
                print(f"  Roll back: resume_checkpoint_{epoch:03d}.pt")

        results_logger.log_epoch(
            epoch=epoch, train_loss=train_loss.avg, val_loss=val_loss.avg,
            train_metrics=t_avgs, val_metrics=v_avgs,
        )
        plot_training_dashboard(results_logger.history, epoch + 1, cfg.dashboard_dir)

        if not epoch_is_nan:
            if val_loss.avg < best_loss:
                best_loss      = val_loss.avg
                best_epoch     = epoch
                num_bad_epochs = 0
                torch.save(raw_model.state_dict(),
                           os.path.join(cfg.checkpoint_dir, "best.pt"))
                print(f"  Best model saved  (val_loss={best_loss:.4f})")
            else:
                num_bad_epochs += 1
           
            lr_scheduler.step()

        try:
            save_checkpoint(
                epoch=epoch, model=raw_model, optimizer=optimizer,
                lr_scheduler=lr_scheduler, best_loss=best_loss,
                num_bad_epochs=num_bad_epochs, best_epoch=best_epoch,
                path=os.path.join(cfg.checkpoint_dir,
                                  f"resume_checkpoint_{epoch+1}.pt"),
                cfg=cfg,
            )
        except Exception as e:
            print(f"[ERROR] save_checkpoint failed: {e}")
            raise

  
    if is_main:
        torch.save(raw_model.state_dict(),
                   os.path.join(cfg.checkpoint_dir, "last.pt"))
        print(f"\n{'='*72}")
        print(f"  Training complete.")
        print(f"  Seed       : {cfg.seed}")
        print(f"  Best epoch : {best_epoch+1}  |  best val_loss : {best_loss:.4f}")
        print(f"  Metrics    : {cfg.metrics_dir}")
        print(f"  Dashboards : {cfg.dashboard_dir}")
        print(f"{'='*72}")

    if is_ddp:
        dist.destroy_process_group()

    model = optimizer = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()