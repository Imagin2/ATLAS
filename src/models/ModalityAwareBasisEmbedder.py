import math
import torch
import torch.nn as nn
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from config import CFG


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding (1, max_len, d_model).
    Initialised with trunc_normal(std=1/sqrt(d_model)).
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        std     = 1.0 / math.sqrt(d_model)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, mean=0.0, std=std, a=-2 * std, b=2 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]



class ModalityAwareBasisEmbedder(nn.Module):
    _MOD2IDX = {"image": 0, "text": 1}

    def __init__(self, patch_dim: int = 768, vocab_size: int = 8192,
                 d_model: int = 512, d_inter: int = 512,
                 max_seq_len: int = 512, num_latents: int = 64,
                 num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_latents = num_latents

        self.img_proj_1 = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_inter),
        )
        self.txt_proj_1 = nn.Embedding(vocab_size, d_inter)

        self.shared_proj_2 = nn.Sequential(
            nn.Linear(d_inter, d_model),
            nn.LayerNorm(d_model),
        )
        self.content_pos_enc = LearnablePositionalEncoding(d_model, max_len=max_seq_len)

        # one-hot(image/text) -> modality token  (lookup == one_hot @ W)
        self.modality_emb = nn.Embedding(2, d_model)
        # learnable positional encoding added to the modality token
        std = 1.0 / math.sqrt(d_model)
        self.basis_pos = nn.Parameter(torch.empty(1, num_latents, d_model))
        nn.init.trunc_normal_(self.basis_pos, mean=0.0, std=std, a=-2 * std, b=2 * std)

        self.attention    = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, X: torch.Tensor, modality: str) -> torch.Tensor:
        X_proj = self.img_proj_1(X) if modality == "image" else self.txt_proj_1(X)
        X_proj = self.shared_proj_2(X_proj)
        X_proj = self.content_pos_enc(X_proj)

        idx     = torch.tensor(self._MOD2IDX[modality], device=X_proj.device)
        mod_tok = self.modality_emb(idx).view(1, 1, -1)  
        Bmi     = (mod_tok + self.basis_pos).expand(X_proj.size(0), -1, -1)  

        attn_out, _ = self.attention(query=X_proj, key=Bmi, value=Bmi)
        return self.attn_dropout(attn_out) + X_proj


def build_embedder(cfg: CFG) -> nn.Module:
    common = dict(
        patch_dim   = cfg.patch_dim,
        vocab_size  = cfg.vocab_size,
        d_model     = cfg.d_model,
        d_inter     = cfg.d_inter,
        max_seq_len = cfg.max_seq_len,
        num_heads   = cfg.embedder_heads,
        dropout     = cfg.embedder_dropout,
        num_latents = cfg.num_latents,
    )
    if cfg.embedder == "learnable":
        return ModalityAwareBasisEmbedder(**common)
    else:
        raise ValueError(f"Unknown embedder '{cfg.embedder}'")