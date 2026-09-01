import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import math
from dataclasses import dataclass
from typing import Optional, Union
import albumentations as A
import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from config import CFG
from src.utils.metrics import compute_metrics
from src.models.ModalityAwareBasisEmbedder import build_embedder
from src.dataset.dataset import PatchExtractor
from PIL import Image

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


class AttentionPooling(nn.Module):
    """Learned attention pooling over content tokens."""
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        scores = self.score(x).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)
        pooled       = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        return pooled, attn_weights


class UniversalProjectionEncoder(nn.Module):
    """
    Shared Pre-LN transformer encoder for image and text.
    Output norm applied after the transformer (encoder-exit norm).
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
        B        = EX.size(0)
        device   = EX.device
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


class Backbone(nn.Module):
    """
    Unified image + text encoder.
    IMAGE : (B,3,H,W)  -> PatchExtractor -> ModalityAwareBasisEmbedder -> UniversalEncoder
    TEXT  : (B,seq)    -> ModalityAwareBasisEmbedder -> UniversalEncoder
    """
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg      = cfg
        self.device   = cfg.device
        self.embedder = cfg.embedder

        self.patch_extractor   = PatchExtractor(
            image_size=cfg.image_size, patch_size=cfg.patch_size, channels=cfg.channels)
        self.modality_embedder = build_embedder(cfg)
        self.universal_encoder = UniversalProjectionEncoder(
            d_model=cfg.d_model, num_head=cfg.num_head,
            num_layers=cfg.num_layers, dropout_rate=cfg.dropout_rate,
            pooling_type=cfg.pooling_type,
        )
        self.image_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model,cfg.d_model),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model,cfg.d_model),
        )

  

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        if image_path.startswith("http"):
            response  = requests.get(image_path, timeout=15)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            return cv2.cvtColor(cv2.imdecode(img_array, cv2.IMREAD_COLOR),
                                cv2.COLOR_BGR2RGB)
        return np.array(Image.open(image_path).convert("RGB"))

    def _image_transform(self) -> A.Compose:
        return A.Compose([
            A.Resize(self.cfg.image_size, self.cfg.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _encode_image_backbone(self, images: torch.Tensor) -> UniversalProjectionOutput:
        patches = self.patch_extractor(images)
        B, S, _ = patches.shape
        img_EX = self.modality_embedder(patches, modality="image")
        return self.universal_encoder(img_EX, modality="image", attention_mask=None)

    def _encode_text_backbone(self, input_ids: torch.Tensor,
                              attention_mask: torch.Tensor) -> UniversalProjectionOutput:
        B, S = input_ids.shape
        txt_EX = self.modality_embedder(input_ids, modality="text")
        return self.universal_encoder(txt_EX, modality="text",
                                      attention_mask=attention_mask)

    @staticmethod
    def _select_return(out: ModalityEncodingOutput,
                       return_type: str) -> Union[ModalityEncodingOutput, torch.Tensor]:
        mapping = {
            "all": out, "feature": out.feature, "embedding": out.embedding,
            "sequence": out.sequence, "cls": out.cls_output,
            "attn": out.attn_pool_output, "mean": out.mean_output,
        }
        if return_type not in mapping:
            raise ValueError(f"Unknown return_type: {return_type!r}")
        return mapping[return_type]

    def encode_image(self, images: Optional[torch.Tensor] = None,
                     image_paths: Optional[list] = None,
                     return_type: str = "embedding") -> Union[ModalityEncodingOutput, torch.Tensor]:
        if image_paths is not None:
            tf     = self._image_transform()
            images = torch.stack([
                tf(image=self._load_image(p))["image"].float() for p in image_paths
            ]).to(self.device)
        elif images is None:
            raise ValueError("Provide either images or image_paths.")
        images = images.to(self.device)
        enc    = self._encode_image_backbone(images)
        out    = ModalityEncodingOutput(
            feature=enc.pooler_output,
            embedding=F.normalize(self.image_proj(enc.pooler_output), p=2, dim=-1),
            sequence=enc.last_hidden_state,
            cls_output=enc.cls_output,
            attn_pool_output=enc.attn_pool_output,
            mean_output=enc.mean_output,
            attn_weights=enc.attn_weights,
        )
        return self._select_return(out, return_type)

    def encode_text(self, tokenizer=None, texts: Optional[list] = None,
                    input_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    max_length: Optional[int] = None,
                    return_type: str = "embedding") -> Union[ModalityEncodingOutput, torch.Tensor]:
        if texts is not None:
            if tokenizer is None:
                raise ValueError("tokenizer required when texts is provided.")
            enc_out        = tokenizer(texts, padding=True, truncation=True,
                                       max_length=max_length or self.cfg.max_length,
                                       return_tensors="pt")
            input_ids      = enc_out["input_ids"]
            attention_mask = enc_out["attention_mask"]
        elif input_ids is None or attention_mask is None:
            raise ValueError("Provide texts+tokenizer or input_ids+attention_mask.")

        input_ids      = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        enc = self._encode_text_backbone(input_ids, attention_mask)
        out = ModalityEncodingOutput(
            feature=enc.pooler_output,
            embedding=F.normalize(self.text_proj(enc.pooler_output), p=2, dim=-1),
            sequence=enc.last_hidden_state,
            cls_output=enc.cls_output,
            attn_pool_output=enc.attn_pool_output,
            mean_output=enc.mean_output,
            attn_weights=enc.attn_weights,
        )
        return self._select_return(out, return_type)

    def get_image_feature(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images=images, return_type="feature")

    def get_text_feature(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode_text(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_type="feature")



class RetrievalPretrainModel(nn.Module):
    """
    Contrastive pre-training wrapper around Backbone.
    
    """
    def __init__(self, cfg: CFG, backbone: Backbone,
                 compute_metrics_fn):
        super().__init__()
        self.cfg                = cfg
        self.backbone           = backbone
        self.compute_metrics_fn = compute_metrics_fn
        self.loss_type          = cfg.loss_type.lower()

        if self.loss_type == "clip_loss":
            if cfg.temperature_mode == "learned":
                self.temperature = nn.Parameter(
                    torch.tensor(math.log(1.0 / cfg.temperature_value))
                )
                self._temp_min = cfg.temperature_min
                self._temp_max = cfg.temperature_max
            else:
                self.register_buffer(
                    "temperature",
                    torch.tensor(math.log(1.0 / cfg.temperature_value))
                )

    def _get_temperature(self) -> torch.Tensor:
        if self.cfg.temperature_mode == "learned":
            return torch.exp(
                torch.clamp(self.temperature, self._temp_min, self._temp_max)
            )
        return torch.exp(self.temperature)

    
    def forward(self, batch: dict, return_metrics: bool = False,
               ):
        
        images         = batch["image"].to(self.cfg.device, non_blocking=True)
        input_ids      = batch["input_ids"].to(self.cfg.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.cfg.device, non_blocking=True)
        B              = images.size(0)

        img_out = self.backbone.encode_image(images=images, return_type="embedding")
        txt_out = self.backbone.encode_text(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            return_type="embedding")

        if self.loss_type == "clip_loss":
            t      = self._get_temperature()
            logits = (img_out @ txt_out.T) * t
            labels = torch.arange(logits.size(0), device=images.device)
            loss   = 0.5 * (F.cross_entropy(logits, labels) +
                            F.cross_entropy(logits.T, labels))
            
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

        if not return_metrics:
            return loss

        metrics         = self.compute_metrics_fn(
            img_out, txt_out,
           
        )
        metrics["loss"] = loss.item()
        return loss, metrics


def build_stage1_model(cfg: CFG) -> RetrievalPretrainModel:
    backbone = Backbone(cfg)
    return RetrievalPretrainModel(cfg, backbone=backbone,
                                  compute_metrics_fn=compute_metrics).to(cfg.device)