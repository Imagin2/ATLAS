
import os
import platform
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
import albumentations as A
from config import CFG
from albumentations.pytorch import ToTensorV2
import cv2
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
def make_pairs(annotation_json_file: str, image_dir: str, cfg: CFG,
               max_captions: int = 3, train: bool = True):
    """
    Build (image_path, caption) pairs from COCO + Flickr30k / TextCaps.
    Returns:
    train=True  : (coco_pairs, flickr_pairs, textcaps_pairs)
    train=False : (coco_pairs, None, textcaps_pairs)
    from:https://github.com/b-faye/OneEncoder
    """
    coco      = COCO(annotation_json_file)
    image_ids = list(coco.imgs.keys())

    coco_pairs = []
    for image_id in image_ids:
        num_caption = 0
        image_info  = coco.loadImgs(image_id)[0]
        image_path  = os.path.join(image_dir, image_info["file_name"])
        ann_ids     = coco.getAnnIds(image_id)
        for ann in coco.loadAnns(ann_ids):
            if "caption" in ann and num_caption < max_captions:
                coco_pairs.append((image_path, ann["caption"]))
                num_caption += 1

    if train:
        flickr_pairs = []
        flickr_df    = pd.read_csv(
            os.path.join(cfg.data_directory, "flickr30k_images", "results.csv"),
            delimiter="|",
        )
        flickr_df.loc[flickr_df[" comment"].isna(), " comment"] = (
            flickr_df.loc[flickr_df[" comment"].isna(), " comment_number"]
        )
        for _, row in flickr_df.iterrows():
            img_path = os.path.join(
                cfg.data_directory, "flickr30k_images/flickr30k_images",
                row["image_name"],
            )
            flickr_pairs.append((img_path, row[" comment"]))

        textcaps_pairs = []
        tc_df = pd.read_csv(
            os.path.join(cfg.data_directory, "TextCaps/textcaps_train.csv")
        )
        for _, row in tc_df.iterrows():
            img_path = os.path.join(
                cfg.data_directory, "TextCaps/images_zip/images/train_val",
                row["image"],
            )
            textcaps_pairs.append((img_path, row["answer"]))

        return coco_pairs, flickr_pairs, textcaps_pairs

    else:
        textcaps_pairs = []
        tc_df = pd.read_csv(
            os.path.join(cfg.data_directory, "TextCaps/textcaps_val.csv")
        )
        for _, row in tc_df.iterrows():
            img_path = os.path.join(
                cfg.data_directory, "TextCaps/images_zip/images/train_val",
                row["image"],
            )
            textcaps_pairs.append((img_path, row["answer"]))

        return coco_pairs, None, textcaps_pairs


def _make_randaugment_equiv() -> list:
    """Version-aware RandAugment substitute."""
    import albumentations as _A
    if hasattr(_A, "RandAugment"):
        return [_A.RandAugment(num_ops=2, magnitude=9, p=0.7)]
    pool = [
        _A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
        _A.Equalize(p=1.0),
        _A.Sharpen(alpha=(0.1, 0.4), lightness=(0.7, 1.3), p=1.0),
        _A.Posterize(num_bits=(4, 8), p=1.0),
        _A.Solarize(p=1.0),
        _A.Rotate(limit=30, p=1.0),
        _A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=1.0),
        _A.AutoContrast(p=1.0) if hasattr(_A, "AutoContrast") else _A.CLAHE(p=1.0),
    ]
    if hasattr(_A, "SomeOf"):
        return [_A.SomeOf(pool, n=2, p=0.7)]
    return [
        _A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.4),
        _A.Equalize(p=0.2),
        _A.Sharpen(p=0.2),
        _A.Rotate(limit=20, p=0.3),
    ]


def _make_coarse_dropout() -> object:
    """
    Version-aware CoarseDropout 
    """
    import albumentations as _A
    try:
        return _A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.25,
        )
    except TypeError:
        return _A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, min_height=8, min_width=8,
            p=0.25,
        )


def get_transforms(cfg: CFG, mode: str = "train") -> A.Compose:
    """
    Albumentations pipeline.
    
    """
    if mode == "train":
        return A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.HorizontalFlip(p=0.5),
            *_make_randaugment_equiv(),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.ToGray(p=0.1),
            _make_coarse_dropout(),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(cfg.image_size, cfg.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class DataGenerator(torch.utils.data.Dataset):
    
    def __init__(self, images, captions, tokenizer, transformer, cfg: CFG,
                 caption_drop_prob: float = 0.0, cache_images: bool = True):
        super().__init__()
        self.transformer       = transformer
        self.images            = images
        self.captions          = captions
        self.cfg               = cfg
        self.caption_drop_prob = caption_drop_prob
        self.pad_id            = getattr(tokenizer, "pad_id",
                                         getattr(tokenizer, "pad_token_id", 0))
        self.bos_id            = getattr(tokenizer, "bos_id",
                                         getattr(tokenizer, "cls_token_id", 1))
        self.eos_id            = getattr(tokenizer, "eos_id",
                                         getattr(tokenizer, "sep_token_id", 2))
        self.tokenized_captions = tokenizer(
            list(self.captions), padding=True,
            truncation=True, max_length=cfg.max_length,
        )

        # Pre-load unique images into RAM
        self.image_cache: dict = {}
        if cache_images:
            unique_paths = list(dict.fromkeys(images))
            is_main      = getattr(cfg, "is_main", True)
            rank_str     = f"[rank{cfg.local_rank}] " if cfg.local_rank >= 0 else ""
            print(f"  {rank_str}[DataGenerator] Pre-loading {len(unique_paths)} images ...")
            for path in tqdm(unique_paths, disable=not is_main, desc=f"  {rank_str}cache"):
                img = cv2.imread(path)
                if img is not None:
                    img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w  = img.shape[:2]
                    scale = 256 / min(h, w)
                    if scale < 1.0:
                        img = cv2.resize(img,
                                         (int(w * scale), int(h * scale)),
                                         interpolation=cv2.INTER_LINEAR)
                    self.image_cache[path] = img
            mb = sum(a.nbytes for a in self.image_cache.values()) / 1024 ** 2
            print(f"  {rank_str}[DataGenerator] Cache ready: "
                  f"{len(self.image_cache)} images, {mb:.0f} MB RAM")

    def __len__(self):
        return len(self.captions)

    def _apply_caption_dropout(self, ids: torch.Tensor,
                                mask: torch.Tensor) -> tuple:
        """Replace non-special tokens with pad_id at random."""
        if self.caption_drop_prob <= 0.0:
            return ids, mask
        special   = (ids == self.pad_id) | (ids == self.bos_id) | (ids == self.eos_id)
        drop      = torch.bernoulli(
            torch.full(ids.shape, self.caption_drop_prob)
        ).bool() & ~special
        ids       = ids.clone()
        ids[drop] = self.pad_id
        return ids, mask

    def __getitem__(self, index):
        item = {k: torch.tensor(v[index])
                for k, v in self.tokenized_captions.items()}

        item["input_ids"], item["attention_mask"] = self._apply_caption_dropout(
            item["input_ids"], item["attention_mask"]
        )

        # Serve from RAM cache
        path  = self.images[index]
        image = self.image_cache.get(path)
        if image is None:
            image = cv2.imread(path)
            if image is None:
                image = np.zeros(
                    (self.cfg.image_size, self.cfg.image_size, 3), dtype=np.uint8
                )
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        item["image"]   = self.transformer(image=image)["image"].float()
        item["caption"] = self.captions[index]
        return item


def build_loader(dataframe, tokenizer, cfg: CFG,
                 mode: str = "train") -> torch.utils.data.DataLoader:
  
    
    is_ddp = dist.is_available() and dist.is_initialized()

    if is_ddp:
        world = dist.get_world_size()
        rank  = dist.get_rank()
        
        dataframe = dataframe[rank::world]

    transformer = get_transforms(cfg=cfg, mode=mode)
    drop_prob   = getattr(cfg, "caption_drop_prob", 0.0) if mode == "train" else 0.0

    dataset = DataGenerator(
        images            = [img     for img, _       in dataframe],
        captions          = [caption for _,   caption in dataframe],
        tokenizer         = tokenizer,
        transformer       = transformer,
        cfg               = cfg,
        caption_drop_prob = drop_prob,
        cache_images      = True,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size         = cfg.batch_size,
        num_workers        = cfg.num_workers,
        shuffle            = (mode == "train"),   
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 4,
    )


class PatchExtractor(nn.Module):
    """
    Splits (B, C, H, W) images into flat patches and layer-normalises them.
    Output: (B, num_patches, patch_dim)  patch_dim = C * P * P
    """
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 channels: int = 3):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_dim = channels * patch_size * patch_size
        self.rearrange = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size, p2=patch_size,
        )
        self.norm = nn.LayerNorm(self.patch_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.norm(self.rearrange(images))
