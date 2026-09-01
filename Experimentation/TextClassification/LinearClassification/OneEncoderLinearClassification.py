import argparse
import torch

from Experimentation.OneEncoder import OneEncoder
import probe_commun as pc


def _lock(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def load_oneencoder(model_path, device):
    model = OneEncoder(device=device).to(device)
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
    _lock(model)
    return model


def make_oneencoder_tokenize(tokenizer, max_length=128):
    """Use the backbone's own BERT tokenizer to produce (ids, mask)."""
    def _tok(texts):
        out = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        return out["input_ids"].long(), out["attention_mask"].long()
    return _tok


def make_extract_fn(model):
    def _extract(input_ids, attention_mask):
        with torch.no_grad():
            text_feats = model.text_encoder([input_ids, attention_mask])
            mod_tok    = model.modality_token_encoder("text")
            out        = model.universal_projection_encoder([text_feats, mod_tok])
            return out.mean_output.float()
    return _extract


def parse_args():
    p = argparse.ArgumentParser(description="OneEncoder text linear probe")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="./text_probe_results_oneencoder")
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model  = load_oneencoder(args.model_path, device)

    adapter = pc.TextBackboneAdapter(
        module=model,
        extract_fn=make_extract_fn(model),
        tokenize_fn=make_oneencoder_tokenize(model.tokenizer, args.max_length),
        name="oneencoder",
    )
    pc.run_probes(adapter, args.datasets, args.output_dir,
                  head=args.head, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, lr=args.lr,
                  weight_decay=args.weight_decay, num_workers=args.num_workers,
                  device=device, seed=args.seed, model_path=args.model_path)


if __name__ == "__main__":
    main()