#python ./Experimentation/ImageClassification/LinearClassification/OneEncoderLinearClassification.py --model_path ./Results/best_new.pt --datasets oxfordpets --lr 1e-2
import argparse
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from Experimentation.OneEncoder import OneEncoder 
import probe_commun as pc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


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
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="OneEncoder")
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_root", default=".data")
    p.add_argument("--output_dir", default="./probe_results_oneencoder")
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
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model  = load_oneencoder(args.model_path, device)

    adapter = pc.BackboneAdapter(
        module=model,
        extract_fn=lambda imgs: model.encode_image(image_tensors=imgs,
                                                    outputs="mean").float(),
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD, name="oneencoder",
    )
    pc.run_probes(adapter, args.datasets, args.data_root, args.output_dir,
                  head=args.head, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, lr=args.lr,
                  weight_decay=args.weight_decay, num_workers=args.num_workers,
                  device=device, seed=args.seed, model_path=args.model_path)


if __name__ == "__main__":
    main()