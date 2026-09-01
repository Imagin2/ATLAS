import argparse
import torch

from Experimentation.Clip import CLIPModel, SimpleTokenizer, compute_metrics
import probe_commun as pc


class EvalCFG:
    def __init__(self, **kw):
        self.device = torch.device(kw.pop("device", None)
                                   or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.local_rank = -1
        self.is_main = True
        for k, v in kw.items():
            setattr(self, k, v)
        self.patch_dim   = self.channels * self.patch_size ** 2
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.checkpoint_dir = self.metrics_dir = self.dashboard_dir = self.embeddings_dir = "./results"
        self.run_name = "clip_text_probe"


def clip_pre_proj_text_feature(text_encoder, input_ids):
    """Replicate CLIPTextEncoder.forward up to (not incl.) text_projection."""
    enc = text_encoder
    x = enc.token_embedding(input_ids).type(enc.dtype)
    x = x + enc.positional_embedding.type(enc.dtype)
    x = x.permute(1, 0, 2)
    x = enc.transformer(x)
    x = x.permute(1, 0, 2)
    x = enc.ln_final(x).type(enc.dtype)
    feat = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)]  
    return feat.float()


def make_clip_tokenize(tokenizer, context_length=77):
    def _tok(texts):
        out = tokenizer(list(texts), context_length=context_length,
                        truncate=True, return_tensors="pt")
        ids  = out["input_ids"]                 
        mask = (ids != 0).long()                 
        return ids, mask
    return _tok


def load_clip(model_path, cfg, device):
    model = CLIPModel(cfg, compute_metrics).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  [WARN] missing ({len(missing)}): {missing[:5]} ...")
    if unexpected: print(f"  [WARN] unexpected ({len(unexpected)}): {unexpected[:5]} ...")
    if not missing and not unexpected:
        print("  [OK] state dict loaded (strict).")
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="CLIP")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="./text_probe_results_clip")
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
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
    tokenizer = SimpleTokenizer()

    adapter = pc.TextBackboneAdapter(
        module=model,
        extract_fn=lambda ids, mask: clip_pre_proj_text_feature(model.text_encoder, ids),
        tokenize_fn=make_clip_tokenize(tokenizer, args.context_length),
        name="clip",
    )
    pc.run_probes(adapter, args.datasets, args.output_dir,
                  head=args.head, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, lr=args.lr,
                  weight_decay=args.weight_decay, num_workers=args.num_workers,
                  device=device, seed=args.seed, model_path=args.model_path)


if __name__ == "__main__":
    main()