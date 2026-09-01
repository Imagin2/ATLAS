import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import argparse
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Experimentation.Clip import CLIPModel, compute_metrics
import probe_commun as pc

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)



class EvalCFG:
    def __init__(self, **kw):
        self.device     = torch.device(kw.pop("device", None)
                                        or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.local_rank = -1
        self.is_main    = True
        for k, v in kw.items():
            setattr(self, k, v)
        self.patch_dim   = self.channels * self.patch_size ** 2
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.checkpoint_dir = "./results"
        self.metrics_dir = self.dashboard_dir = self.embeddings_dir = "./results"
        self.run_name = "clip_probe"


def clip_pre_proj_feature(image_encoder, images):
    enc = image_encoder
    x = enc.conv1(images.type(enc.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = enc.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + enc.positional_embedding.to(x.dtype)
    x = enc.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = enc.transformer(x)
    x = x.permute(1, 0, 2)
    x = enc.ln_post(x[:, 0, :])         
    return x.float()


def load_clip(model_path, cfg, device):
    model = CLIPModel(cfg, compute_metrics).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  [WARN] missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected: print(f"  [WARN] unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    if not missing and not unexpected:
        print("  [OK] state dict loaded (strict).")
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="CLIP")
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--output_dir", default="./probe_results_clip")
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--head", default="mlp", choices=["linear", "mlp"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_head", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--vocab_size", type=int, default=49408)
    p.add_argument("--context_length", type=int, default=77)
    p.add_argument("--temperature_mode", default="learned", choices=["learned", "fixed"])
    p.add_argument("--temperature_value", type=float, default=0.07)
    p.add_argument("--temperature_min", type=float, default=1.5)
    p.add_argument("--temperature_max", type=float, default=4.0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cfg    = EvalCFG(**vars(args)); cfg.device = device
    model  = load_clip(args.model_path, cfg, device)

    adapter = pc.BackboneAdapter(
        module=model,
        extract_fn=lambda imgs: clip_pre_proj_feature(model.image_encoder, imgs),
        norm_mean=CLIP_MEAN, norm_std=CLIP_STD, name="clip",
    )
    pc.run_probes(adapter, args.datasets, args.data_root, args.output_dir,
                  head=args.head, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, lr=args.lr,
                  weight_decay=args.weight_decay, num_workers=args.num_workers,
                  device=device, seed=args.seed, model_path=args.model_path)


if __name__ == "__main__":
    main()