import math, os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from transformers import BertTokenizer
import sentencepiece as spm
import cv2
from PIL import Image
from tqdm import tqdm

from sklearn.decomposition import PCA as skPCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

torch.manual_seed(0)
np.random.seed(0)


MABE_CKPT   = ("./Results/"
              "best.pt")

SEP_CKPT   = "./Results/best_OneEncoder.pt"

CLIP_ARCH       = "ViT-B-32"
CLIP_PRETRAINED = "openai"

DATA_DIR   = "./data"
VAL_ANN    = os.path.join(DATA_DIR, "coco-image-caption",
                          "annotations_trainval2014", "annotations",
                          "captions_val2014.json")
VAL_IMG    = os.path.join(DATA_DIR, "coco-image-caption", "val2014")
SPM_MODEL  = ("./"
              "spm.model")
BERT_NAME  = "bert-base-uncased"

N_SAMPLES   = 512  
BATCH_SIZE  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1a. MABE architecture
class LearnablePositionalEncoding_MABE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        std = 1.0 / math.sqrt(d_model)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=std)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MABEEmbedder(nn.Module):
    
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
        self.content_pos_enc = LearnablePositionalEncoding_MABE(d_model, max_len=max_seq_len)

      
        self.modality_emb = nn.Embedding(2, d_model)
       
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



class _AttentionPool(nn.Module):
    
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
    def forward(self, x, attention_mask=None):
        scores = self.score(x).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (x * weights.unsqueeze(-1)).sum(1), weights


class MABESharedEncoder(nn.Module):
    
    def __init__(self, d_model=512, num_head=4, num_layers=4,
                 dropout=0.1, pooling_type="attn"):
        super().__init__()
        self.pooling_type = pooling_type
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_head,
            dim_feedforward=d_model*4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_norm    = nn.LayerNorm(d_model)
        self.cls_image      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_text       = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attention_pool = _AttentionPool(d_model, dropout)

    def forward(self, EX, modality, attention_mask=None):
        B      = EX.size(0)
        cls    = (self.cls_image if modality == "image"
                  else self.cls_text).expand(B, 1, -1)
        x      = torch.cat([cls, EX], dim=1)
        if attention_mask is not None:
            cls_m    = torch.ones(B, 1, dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            full_m   = torch.cat([cls_m, attention_mask], dim=1)
            pad_mask = ~full_m.bool()
        else:
            pad_mask = None
        encoded  = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        encoded  = self.output_norm(encoded)
        content  = encoded[:, 1:, :]
        pooled, _ = self.attention_pool(content, attention_mask=attention_mask)
        return pooled   


class MABEPatchExtractor(nn.Module):
    def __init__(self, image_size=224, patch_size=16, channels=3):
        super().__init__()
        self.patch_dim = channels * patch_size * patch_size
        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                                   p1=patch_size, p2=patch_size)
        self.norm = nn.LayerNorm(self.patch_dim)
    def forward(self, images):
        return self.norm(self.rearrange(images))



class LearnablePositionalEncoding_SEP(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout   = nn.Dropout(dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        return self.dropout(x + self.pos_embed(pos))


class SEPProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=256, dropout_rate=0.1):
        super().__init__()
        self.linear_layer1       = nn.Linear(input_dim, projection_dim)
        self.gelu                = nn.GELU()
        self.linear_layer2       = nn.Linear(projection_dim, projection_dim)
        self.dropout             = nn.Dropout(dropout_rate)
        self.normalization_layer = nn.LayerNorm(projection_dim)
    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.gelu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        x = self.normalization_layer(x)
        return x


class SEPImageEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 d_model=512, num_heads=8, num_layers=4,
                 dim_feedforward=2048, dropout_rate=0.1, projection_dim=256):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, d_model,
                                          kernel_size=patch_size, stride=patch_size)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_encoding = LearnablePositionalEncoding_SEP(
            d_model, max_len=num_patches + 1, dropout=dropout_rate)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout_rate,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.projection_head = SEPProjectionHead(d_model, projection_dim, dropout_rate)

    def forward(self, images):
        B   = images.size(0)
        x   = self.patch_embedding(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_encoding(x)
        x   = self.transformer(x)
        return self.projection_head(x)


class SEPTextEncoder(nn.Module):
    def __init__(self, vocab_size=30522, max_seq_len=128,
                 d_model=512, num_heads=8, num_layers=4,
                 dim_feedforward=2048, dropout_rate=0.1, projection_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.trunc_normal_(self.token_embedding.weight, std=0.02)
        self.pos_encoding = LearnablePositionalEncoding_SEP(
            d_model, max_len=max_seq_len, dropout=dropout_rate)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout_rate,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.projection_head = SEPProjectionHead(d_model, projection_dim, dropout_rate)

    def forward(self, input_ids, attention_mask):
        x        = self.token_embedding(input_ids)
        x        = self.pos_encoding(x)
        pad_mask = (attention_mask == 0)
        x        = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.projection_head(x)


class SEPModalityTokenEncoder(nn.Module):
    def __init__(self, projection_dim=256, token_size=1, token_dim=3):
        super().__init__()
        self.image_token = nn.Parameter(torch.randn(token_size, token_dim))
        self.text_token  = nn.Parameter(torch.randn(token_size, token_dim))
        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, 64), nn.ReLU(),
            nn.Linear(64, 128),       nn.ReLU(),
            nn.Linear(128, projection_dim), nn.LayerNorm(projection_dim))
    def forward(self, modality_type):
        token = self.image_token if modality_type == "image" else self.text_token
        return self.token_projection(token)


class SEPUniversalEncoder(nn.Module):
    def __init__(self, input_dim=256, num_head=4, num_layers=1):
        super().__init__()
        self.transformer_encoder_block = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_block, num_layers=num_layers)
        self.layer_normalization = nn.LayerNorm(input_dim)
    def forward(self, x, tokens):
        tokens  = tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        out     = x + tokens
        out     = self.layer_normalization(out)
        out     = self.transformer_encoder(out)
        out     = out + (x + tokens)
        return out



def load_MABE(ckpt_path, device):
    state = torch.load(ckpt_path, map_location="cpu")
    backbone_state = {
        k[len("backbone."):]: v
        for k, v in state.items()
        if k.startswith("backbone.")
    }

    patch_ext = MABEPatchExtractor().eval()
    embedder  = MABEEmbedder(
        patch_dim=768, vocab_size=8192, d_model=512, d_inter=512,
        max_seq_len=512, num_latents=64, num_heads=1, dropout=0.1).eval()
    encoder   = MABESharedEncoder(
        d_model=512, num_head=4, num_layers=4,
        dropout=0.1, pooling_type="attn").eval()
    img_proj = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512)).eval()
    txt_proj = nn.Sequential(nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 512)).eval()

    def load_prefix(module, prefix):
        sub = {k[len(prefix):]: v for k, v in backbone_state.items()
               if k.startswith(prefix)}
        missing, unexpected = module.load_state_dict(sub, strict=False)
        param_keys = {n for n, _ in module.named_parameters()}
        truly_missing = [k for k in missing if k in param_keys]
        if truly_missing:
            raise RuntimeError(f"Missing parameter keys in {prefix}: {truly_missing}")

    load_prefix(patch_ext, "patch_extractor.")
    load_prefix(embedder,  "modality_embedder.")
    load_prefix(encoder,   "universal_encoder.")
    load_prefix(img_proj,  "image_proj.")
    load_prefix(txt_proj,  "text_proj.")

    return (patch_ext.to(device), embedder.to(device),
            encoder.to(device), img_proj.to(device), txt_proj.to(device))


def load_sep(ckpt_path, device):
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw["model_state"] if "model_state" in raw else raw

    img_enc  = SEPImageEncoder().eval()
    txt_enc  = SEPTextEncoder().eval()
    mod_tok  = SEPModalityTokenEncoder().eval()
    up_enc   = SEPUniversalEncoder().eval()

    def load_prefix(module, prefix):
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        module.load_state_dict(sub, strict=True)

    load_prefix(img_enc,  "image_encoder.")
    load_prefix(txt_enc,  "text_encoder.")
    load_prefix(mod_tok,  "modality_token_encoder.")
    load_prefix(up_enc,   "universal_projection_encoder.")

    return img_enc.to(device), txt_enc.to(device), mod_tok.to(device), up_enc.to(device)




# 3.  DATA LOADING  (real COCO val images + captions)

def build_val_pairs(ann_file, img_dir, n=N_SAMPLES):
    coco    = COCO(ann_file)
    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)
    pairs = []
    for iid in img_ids:
        info    = coco.loadImgs(iid)[0]
        path    = os.path.join(img_dir, info["file_name"])
        ann_ids = coco.getAnnIds(iid)
        anns    = coco.loadAnns(ann_ids)
        if not anns or not os.path.exists(path):
            continue
        pairs.append((path, anns[0]["caption"]))
        if len(pairs) >= n:
            print("length of pairs", len(pairs))
            break
    return pairs


def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_image_tensor(path, transform):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(image=img)["image"].float()


# 4.  COLLECTION  

@torch.no_grad()
def collect_MABE(pairs, patch_ext, embedder, encoder,
                img_proj, txt_proj, spm_tokenizer, device):
    """Returns ("img"|"txt") -> np.ndarray (N, proj_dim) Stage-2 pooled vectors."""
    transform = get_val_transform()
    out = {"img": [], "txt": []}

    for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc="MABE collect"):
        batch_pairs = pairs[i:i+BATCH_SIZE]
        imgs = torch.stack([load_image_tensor(p, transform)
                            for p, _ in batch_pairs]).to(device)
        patches  = patch_ext(imgs)
        s1_img   = embedder(patches, "image")
        s2_img   = encoder(s1_img, "image", attention_mask=None)
        s2_img_n = F.normalize(img_proj(s2_img), p=2, dim=-1)
        out["img"].append(s2_img_n.cpu())

        captions = [c for _, c in batch_pairs]
        ids_list = []
        for cap in captions:
            ids = ([spm_tokenizer.bos_id()]
                   + spm_tokenizer.encode(cap, out_type=int)
                   + [spm_tokenizer.eos_id()])
            ids_list.append(ids[:160])

        max_len = max(len(x) for x in ids_list)
        input_ids = torch.zeros(len(ids_list), max_len, dtype=torch.long)
        attn_mask = torch.zeros(len(ids_list), max_len, dtype=torch.long)
        for j, ids in enumerate(ids_list):
            input_ids[j, :len(ids)] = torch.tensor(ids)
            attn_mask[j, :len(ids)] = 1
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        s1_txt   = embedder(input_ids, "text")
        s2_txt   = encoder(s1_txt, "text", attention_mask=attn_mask)
        s2_txt_n = F.normalize(txt_proj(s2_txt), p=2, dim=-1)
        out["txt"].append(s2_txt_n.cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in out.items()}


@torch.no_grad()
def collect_sep(pairs, img_enc, txt_enc, mod_tok, up_enc,
                bert_tokenizer, device):
    """Returns ("img"|"txt") -> np.ndarray (N, proj_dim) Stage-2 pooled vectors."""
    transform = get_val_transform()
    out = {"img": [], "txt": []}

    for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc="SEP collect"):
        batch_pairs = pairs[i:i+BATCH_SIZE]
        captions    = [c for _, c in batch_pairs]

        imgs  = torch.stack([load_image_tensor(p, transform)
                             for p, _ in batch_pairs]).to(device)
        img_f = img_enc(imgs)
        img_t = mod_tok("image")
        img_u = up_enc(img_f, img_t)
        img_s2 = img_u.mean(dim=1)
        out["img"].append(img_s2.cpu())

        enc = bert_tokenizer(captions, padding=True, truncation=True,
                             max_length=128, return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        txt_f = txt_enc(ids, mask)
        txt_t = mod_tok("text")
        txt_u = up_enc(txt_f, txt_t)
        txt_s2 = txt_u.mean(dim=1)
        out["txt"].append(txt_s2.cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in out.items()}



@torch.no_grad()
def collect_clip(pairs, clip_model, clip_preprocess, clip_tokenizer, device):
    """OpenAI CLIP via open_clip. Stage-2 = the pooled image/text embeddings
    that CLIP actually uses for retrieval (its joint space). These are the
    native CLIP features; we L2-normalise downstream like the other models."""
    out = {"img": [], "txt": []}
    for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc="CLIP collect"):
        bp = pairs[i:i+BATCH_SIZE]
        caps = [c for _, c in bp]

        imgs = torch.stack([
            clip_preprocess(Image.open(p).convert("RGB")) for p, _ in bp
        ]).to(device)
        img_feat = clip_model.encode_image(imgs)
        out["img"].append(img_feat.float().cpu())

        toks = clip_tokenizer(caps).to(device)
        txt_feat = clip_model.encode_text(toks)
        out["txt"].append(txt_feat.float().cpu())

    return {k: torch.cat(v, dim=0).numpy() for k, v in out.items()}


def load_clip(device):
    """Load OpenAI CLIP through open_clip. Returns (model, preprocess, tokenizer).
    Raises a clear error with install instructions if open_clip is missing."""
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "open_clip not found. Install with:  pip install open_clip_torch\n"
        ) from e
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
    return model.eval().to(device), preprocess, tokenizer

# 5.  METRICS


def pca_2d(Xi, Xt, subsample=1500):
    """Joint PCA onto the first two principal components (PC1/PC2).

    """
    rng  = np.random.default_rng(0)
    ni   = min(subsample, len(Xi))
    nt   = min(subsample, len(Xt))
    Xi_s = Xi[rng.choice(len(Xi), ni, replace=False)]
    Xt_s = Xt[rng.choice(len(Xt), nt, replace=False)]
    X    = np.vstack([Xi_s, Xt_s])
    pca  = skPCA(n_components=2, svd_solver="full").fit(X)
    pca_12  = skPCA(n_components=12, svd_solver="full").fit(X)
    print("12components", pca_12.explained_variance_ratio_)
    Z    = pca.transform(X)
    return Z[:ni], Z[ni:], pca.explained_variance_ratio_


def centroid_gap(Xi, Xt):
    return float(np.linalg.norm(Xi.mean(0) - Xt.mean(0)))


def full_spectrum(Xi, Xt):
    """
    Joint PCA over the FULL rank of the data. Returns the explained-variance

    """
    X = np.vstack([Xi, Xt])
    n = min(X.shape[0] - 1, X.shape[1])     
    pca = skPCA(n_components=n, svd_solver="full").fit(X)
    return pca.explained_variance_ratio_, pca.explained_variance_  


def participation_ratio(eigvals):
    """PR = (Σλ)² / Σλ²  the 'effective number of dimensions'.
    PR = 1 if one axis carries all variance (max anisotropy);
    PR = D if variance is uniform across D dims (perfect isotropy).
    Computed over the FULL eigenvalue spectrum."""
    lam = np.asarray(eigvals, dtype=np.float64)
    s1 = lam.sum()
    s2 = (lam ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def silhouette_modality(Zi, Zt):
    """Silhouette score treating image vs text as the two clusters,
    computed in the 2-D PCA plane that is actually plotted. Higher =
    more separated clouds; near 0 = well-mixed."""
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return None
    X = np.vstack([Zi, Zt])
    y = np.concatenate([np.zeros(len(Zi)), np.ones(len(Zt))]).astype(int)
    return float(silhouette_score(X, y))


def _l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)





# 6.  FIGURE 

def make_figure(models):

    from matplotlib.lines import Line2D
    try:
        from scipy.stats import gaussian_kde
        _HAVE_KDE = True
    except Exception:
        _HAVE_KDE = False
 
    IMG_C  = "#E07B39"
    TXT_C  = "#2E6FB0"
    GRID_C = "#E6E6E6"
    TAGS   = [f"({chr(97+i)})" for i in range(len(models))]
 
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
        "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 300,
    })
 
    panels = []
    for m in models:
        data = m["data"]
      
        Xi = _l2norm(data["img"])
        Xt = _l2norm(data["txt"])
        Zi, Zt, evr = pca_2d(Xi, Xt)
        gap = centroid_gap(Xi, Xt)
        sil = silhouette_modality(Zi, Zt)
        sil_print = f"{sil:.4f}" if sil is not None else "n/a"
        print(f"[{m['label']}]  PC1+PC2 var={evr[0]*100:.1f}%+{evr[1]*100:.1f}%  "
              f"centroid_gap={gap:.4f}  silhouette={sil_print}")
        xy  = np.vstack([Zi, Zt])
        lim = float(np.percentile(np.abs(xy), 99.5) * 1.15)
        if lim <= 0 or not np.isfinite(lim):
            lim = 1.0
        panels.append(dict(Zi=Zi, Zt=Zt, evr=evr, gap=gap, sil=sil, lim=lim))
 
    n = len(models)
    ncols = n        
    nrows = 1
    fig = plt.figure(figsize=(5.5 * ncols, 5.5))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        left=0.075, right=0.985,
        top=0.89, bottom=0.20,
        wspace=0.22, hspace=0.30)
 
    def kde_contours(ax, Z, color, lim):
        if not _HAVE_KDE or len(Z) < 20:
            return
        try:
            k = gaussian_kde(Z.T)
        except Exception:
            return
        g = 120
        xx, yy = np.meshgrid(np.linspace(-lim, lim, g),
                             np.linspace(-lim, lim, g))
        zz = k(np.vstack([xx.ravel(), yy.ravel()])).reshape(g, g)
        levels = np.linspace(zz.max() * 0.20, zz.max() * 0.95, 4)
        ax.contour(xx, yy, zz, levels=levels, colors=[color],
                   linewidths=0.6, alpha=0.55, zorder=2)
 
    for mi, (m, p) in enumerate(zip(models, panels)):
        r, c = divmod(mi, ncols)
        ax = fig.add_subplot(gs[r, c])
        Zi, Zt, evr, lim = p["Zi"], p["Zt"], p["evr"], p["lim"]
 
        ax.set_axisbelow(True)
        ax.grid(True, color=GRID_C, lw=0.6, zorder=0)
        ax.axhline(0, color="#BBBBBB", lw=0.7, zorder=1)
        ax.axvline(0, color="#BBBBBB", lw=0.7, zorder=1)
 
        kde_contours(ax, Zi, IMG_C, lim)
        kde_contours(ax, Zt, TXT_C, lim)
 
        ax.scatter(Zi[:, 0], Zi[:, 1], c=IMG_C, s=30, alpha=0.55,
                   linewidths=0, rasterized=True, zorder=3)
        ax.scatter(Zt[:, 0], Zt[:, 1], c=TXT_C, s=30, alpha=0.55,
                   linewidths=0, rasterized=True, zorder=3)
 
        ci, ct = Zi.mean(0), Zt.mean(0)
        ax.plot([ci[0], ct[0]], [ci[1], ct[1]], color="#444444",
                lw=1.4, ls=(0, (4, 2)), zorder=5)
        ax.scatter(*ci, marker="*", s=420, c=IMG_C,
                   edgecolors="black", lw=0.9, zorder=6)
        ax.scatter(*ct, marker="*", s=420, c=TXT_C,
                   edgecolors="black", lw=0.9, zorder=6)
 
        ax.set_xlabel(f"PC 1  ({evr[0]*100:.1f}% var.)", fontsize=18)
        ax.set_ylabel(f"PC 2  ({evr[1]*100:.1f}% var.)", fontsize=18)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ticks = np.linspace(-lim, lim, 5)
        dec = 0 if lim >= 10 else (1 if lim >= 1 else 2)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([f"{t:.{dec}f}" for t in ticks])
        ax.set_yticklabels([f"{t:.{dec}f}" for t in ticks])
        ax.tick_params(direction="out", length=3.5)
 
        
        
        ax.set_title(f"{TAGS[mi]}  {m['label']}",
                     fontsize=20, pad=8, fontweight="bold", loc="left")
 
        sil_str = f"{p['sil']:.3f}" if p["sil"] is not None else "n/a"
        ann = (f"centroid gap = {p['gap']:.3f}\n"
               f"silhouette   = {sil_str}")
        ax.text(0.035, 0.035, ann, transform=ax.transAxes,
                fontsize=11, va="bottom", ha="left",
                family="monospace", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#CCCCCC", lw=0.6, alpha=0.95))
 
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=IMG_C,
               markersize=10, label="Image", alpha=0.8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TXT_C,
               markersize=10, label="Text", alpha=0.8),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#777777",
               markeredgecolor="black", markeredgewidth=0.9,
               markersize=14, label="Modality centroid"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=16, frameon=False,
                 bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.4, columnspacing=8)
 
    return fig
 
 





def main():
    print("=" * 60)
    print("Loading data …")
    pairs = build_val_pairs(VAL_ANN, VAL_IMG, N_SAMPLES)
    print(f"  {len(pairs)} pairs loaded from COCO val2014")

    print("Loading tokenizers …")
    spm_tok = spm.SentencePieceProcessor()
    spm_tok.load(SPM_MODEL)
    bert_tok = BertTokenizer.from_pretrained(BERT_NAME)

    models = []

    #1. OpenAI CLIP (pretrained)
    try:
        print("\nLoading OpenAI CLIP via open_clip …")
        clip_model, clip_preprocess, clip_tok = load_clip(DEVICE)
        print("Collecting CLIP activations …")
        data_clip = collect_clip(pairs, clip_model, clip_preprocess,
                                 clip_tok, DEVICE)
        models.append(dict(label="CLIP (ViT-B/32)", data=data_clip))
    except Exception as e:
        print(f"  [skip CLIP] {e}")

    # 2. OneEncoder trained from scratch 
    if SEP_CKPT and os.path.exists(SEP_CKPT):
        try:
            print(f"\nLoading OneEncoder (scratch) …\n  {SEP_CKPT}")
            img_enc, txt_enc, mod_tok, up_enc = load_sep(SEP_CKPT, DEVICE)
            print("Collecting OneEncoder (scratch) activations …")
            data_sep = collect_sep(pairs, img_enc, txt_enc, mod_tok, up_enc,
                                   bert_tok, DEVICE)
            models.append(dict(label="OneEncoder*", data=data_sep))
        except Exception as e:
            print(f"  [skip OneEncoder-scratch] {e}")
    else:
        print(f"\n  [skip OneEncoder-scratch] checkpoint not found at {SEP_CKPT}")

    # 3. MABE (ours, from scratch) 
    print(f"\nLoading MABE checkpoint …\n  {MABE_CKPT}")
    patch_ext, embedder, encoder, img_proj, txt_proj = load_MABE(MABE_CKPT, DEVICE)
    print("Collecting MABE activations …")
    data_MABE = collect_MABE(pairs, patch_ext, embedder, encoder,
                           img_proj, txt_proj, spm_tok, DEVICE)
    models.append(dict(label="Ours", data=data_MABE))

    if not models:
        raise RuntimeError("No models could be loaded.")

    

    # Geometry figure (all models)
    print("\nComputing PCA and building geometry figure …")
    fig = make_figure(models)
    fig.savefig("pca_comparison.pdf", dpi=600, bbox_inches="tight",
                facecolor="white", format="pdf")
    fig.savefig("pca_comparison.png", dpi=400, bbox_inches="tight",
                facecolor="white")
    print("Saved : pca_comparison.pdf / .png")

    
    print("Done.")


if __name__ == "__main__":
    main()