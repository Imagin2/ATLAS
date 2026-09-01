import argparse
import torch

from src.models.model import Backbone
from src.dataset.tokenizer import SentencePieceTokenizer
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
        self.pooling_type = self.pooling_type.lower()
        if self.tokenizer == "bert":
            self.vocab_size = 30522


def build_tokenizer(cfg):
    if cfg.tokenizer == "sentencepiece":
        return SentencePieceTokenizer(cfg.spm_model_path)
    elif cfg.tokenizer == "bert":
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained("bert-base-uncased")
    raise ValueError(f"Unknown tokenizer '{cfg.tokenizer}'")


def make_mabe_tokenize(tokenizer, max_length=160):
    def _tok(texts):
        out = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length)
        ids  = torch.tensor(out["input_ids"], dtype=torch.long)
        mask = torch.tensor(out["attention_mask"], dtype=torch.long)
        return ids, mask
    return _tok


def load_mabe(model_path, cfg, device):
    model = Backbone(cfg).to(device)
    ckpt  = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and any(
            k in ckpt for k in ("backbone_state", "model_state", "state_dict")):
        for key in ("backbone_state", "model_state", "state_dict"):
            if ckpt.get(key) is not None:
                ckpt = ckpt[key]; break
    state = ckpt
    if any(k.startswith("backbone.") for k in state.keys()):
        state = {k[len("backbone."):]: v for k, v in state.items()
                 if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  [WARN] missing ({len(missing)}): {missing[:5]} ...")
    if unexpected: print(f"  [WARN] unexpected ({len(unexpected)}): {unexpected[:5]} ...")
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="MABE")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="./text_probe_results_mabe")
    p.add_argument("--spm_model_path",
                   default="./spm.model")
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
    p.add_argument("--d_inter", type=int, default=512)
    p.add_argument("--d_m", type=int, default=64)
    p.add_argument("--num_head", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--embedder_dropout", type=float, default=0.1)
    p.add_argument("--embedder_heads", type=int, default=1)
    p.add_argument("--embedder", default="learnable", choices=["learnable", "onehot"])
    p.add_argument("--tokenizer", default="sentencepiece", choices=["sentencepiece", "bert"])
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--max_length", type=int, default=160)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--num_latents", type=int, default=64)
    p.add_argument("--num_modalities", type=int, default=2)
    p.add_argument("--pooling_type", default="attn", choices=["cls", "mean", "attn"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cfg    = EvalCFG(**vars(args)); cfg.device = device
    model  = load_mabe(args.model_path, cfg, device)
    tokenizer = build_tokenizer(cfg)

    adapter = pc.TextBackboneAdapter(
        module=model,
        extract_fn=lambda ids, mask: model.encode_text(
            input_ids=ids, attention_mask=mask, return_type="feature").float(),
        tokenize_fn=make_mabe_tokenize(tokenizer, args.max_length),
        name="mabe",
    )
    pc.run_probes(adapter, args.datasets, args.output_dir,
                  head=args.head, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, lr=args.lr,
                  weight_decay=args.weight_decay, num_workers=args.num_workers,
                  device=device, seed=args.seed, model_path=args.model_path)


if __name__ == "__main__":
    main()