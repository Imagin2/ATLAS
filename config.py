import argparse
import os
import torch 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MABE contrastive training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    paths = p.add_argument_group("paths")
    paths.add_argument("--data_directory", type=str,
        default="./data")
    paths.add_argument("--checkpoint_dir", type=str,
        default="./RESULTS")
    paths.add_argument("--spm_model_path", type=str,
        default="./spm.model")
    paths.add_argument("--resume_checkpoint", type=str, default="",
        help="Path to checkpoint to resume from")
    paths.add_argument("--run_name", type=str, default=None,
        help="Override auto-generated run name")

    # Ablation
    abl = p.add_argument_group("ablation")
    abl.add_argument("--tokenizer", type=str, default="sentencepiece",
        choices=["sentencepiece", "bert"])
    abl.add_argument("--embedder", type=str, default="learnable",
       )
    abl.add_argument("--temperature_mode", type=str, default="learned",
        choices=["learned", "fixed"])
    abl.add_argument("--temperature_value", type=float, default=0.07,
        help="Initial temperature")
    abl.add_argument("--temperature_min", type=float, default=1.5,
        help="Lower clamp bound for learned log-temperature (log-space)")
    abl.add_argument("--temperature_max", type=float, default=4.0,
        help="Upper clamp bound for learned log-temperature (log-space)")
   

    # Architecture
    arch = p.add_argument_group("architecture")
    arch.add_argument("--image_size",       type=int,   default=224)
    arch.add_argument("--patch_size",       type=int,   default=16)
    arch.add_argument("--channels",         type=int,   default=3)
    arch.add_argument("--d_model",          type=int,   default=512)
    arch.add_argument("--d_inter",          type=int,   default=512)
    arch.add_argument("--num_head",         type=int,   default=4)
    arch.add_argument("--num_layers",       type=int,   default=4)
    arch.add_argument("--dropout_rate",     type=float, default=0.1)
    arch.add_argument("--embedder_dropout", type=float, default=0.1)
    arch.add_argument("--embedder_heads",   type=int,   default=1)
    arch.add_argument("--vocab_size",       type=int,   default=8192)
    arch.add_argument("--max_length",       type=int,   default=160,
        help="Max token sequence length for text")
    arch.add_argument("--max_seq_len",      type=int,   default=512,
        help="Max sequence length for positional encodings in embedder")
    arch.add_argument("--num_latents",      type=int,   default=64,
        help="Basis size L: K/V vectors in embedder cross-attention")
    arch.add_argument("--num_modalities",   type=int,   default=2)
    arch.add_argument("--pooling_type",     type=str,   default="attn",
        choices=["cls", "mean", "attn"])

    # Training
    train = p.add_argument_group("training")
    train.add_argument("--batch_size",      type=int,   default=64,
        help="Per-GPU batch size.")
    train.add_argument("--num_workers",     type=int,   default=12,
        help="DataLoader workers per GPU.")
    train.add_argument("--lr",              type=float, default=1e-4)
    train.add_argument("--use_amp",         action="store_true")
    train.add_argument("--no_amp",          dest="use_amp", action="store_false")
    p.set_defaults(use_amp=True)
    train.add_argument("--max_grad_norm",   type=float, default=1.0,
        help="Gradient clipping max norm")
    train.add_argument("--weight_decay",    type=float, default=1e-3)
    train.add_argument("--epochs",          type=int,   default=100)
    train.add_argument("--patience",        type=int,   default=5)
    train.add_argument("--factor",          type=float, default=0.8)
    train.add_argument("--max_captions",    type=int,   default=5,
        help="Max captions per image")
    train.add_argument("--caption_drop_prob", type=float, default=0.10,
        help="Per-token masking probability applied to captions during training")
    train.add_argument("--warmup_epochs",   type=int,   default=5,
        help="Linear LR warmup duration before cosine decay")
    train.add_argument("--lr_min",          type=float, default=1e-6,
        help="Minimum LR at the end of cosine decay")
    train.add_argument("--val_size",        type=int,   default=150000)
    train.add_argument("--seed",            type=int,   default=42,
        help="Global random seed.")
    train.add_argument("--deterministic",   action="store_true", default=False)
    train.add_argument("--fast_dev_run",    action="store_true", default=False,
        help="Tessst: cap to 50 train batches, 10 val batches, 2 epochs. "
             )
    train.add_argument("--device",          type=str,   default=None)
    train.add_argument("--loss_type",       type=str,   default="clip_loss")

    return p.parse_args()


class CFG:
    """
    Config built from argparse via CFG(**vars(args)).

    """
    def __init__(self, **kwargs):
        
        raw_device = kwargs.pop("device", None)

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank >= 0:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(
                raw_device if raw_device
                else ("cuda:0" if torch.cuda.is_available() else "cpu")
            )
        self.local_rank = local_rank
        self.is_main    = (local_rank <= 0)   

        for k, v in kwargs.items():
            setattr(self, k, v)

       
        self.patch_dim   = self.channels * self.patch_size ** 2
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Derived data paths
        d = self.data_directory
        self.train_annotation_file = os.path.join(
            d, "coco-image-caption", "annotations_trainval2014",
            "annotations", "captions_train2014.json")
        self.val_annotation_file   = os.path.join(
            d, "coco-image-caption", "annotations_trainval2014",
            "annotations", "captions_val2014.json")
        self.image_dir     = os.path.join(
            d, "coco-image-caption", "train2014", "train2014")
        self.image_dir_val = os.path.join(
            d, "coco-image-caption", "val2014")

        self.pooling_type = self.pooling_type.lower()
        self.loss_type    = self.loss_type.lower()

        
        if self.run_name is None:
            self.run_name = (
                f"tok-{self.tokenizer[:3]}"
                f"_emb-{self.embedder}"
                f"_loss-{self.loss_type}"
                f"_pool-{self.pooling_type}"
                f"_d{self.d_model}L{self.num_layers}H{self.num_head}B{self.batch_size}"
                f"_seed{self.seed}"
                f"_num_latent{self.num_latents}"
                f"_patch_size{self.patch_size}"
            )
        else:
            self.run_name = f"{self.run_name}_seed{self.seed}"

        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.run_name)
        self.metrics_dir    = os.path.join(self.checkpoint_dir, "metrics")
        self.dashboard_dir  = os.path.join(self.checkpoint_dir, "dashboards")
        self.embeddings_dir = os.path.join(self.checkpoint_dir, "embeddings")
        for d_ in [self.metrics_dir, self.dashboard_dir, self.embeddings_dir]:
            os.makedirs(d_, exist_ok=True)

        if self.tokenizer == "bert":
            self.vocab_size = 30522

       

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CFG":
        return cls(**vars(args))

