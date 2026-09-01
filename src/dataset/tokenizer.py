
import os
import sentencepiece as spm
import torch 
from typing import Optional, Union
from config import CFG


def build_corpus(pairs_list: list, output_path: str) -> None:
    """Write all captions to a text file for SentencePiece training."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, caption in pairs_list:
            line = caption.strip()
            if line:
                f.write(line + "\n")
    print(f"  [SPM] Corpus written: {len(pairs_list)} captions  {output_path}")


def train_spm(corpus_path: str, model_prefix: str,
              vocab_size: int = 8192) -> None:
    """Train a SentencePiece BPE model."""
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, pad_piece="<pad>",
        bos_id=1, bos_piece="<s>",
        eos_id=2, eos_piece="</s>",
        unk_id=3, unk_piece="<unk>",
        add_dummy_prefix=False,
    )
    print(f"  [SPM] Model trained  {model_prefix}.model")


class SentencePieceTokenizer:
   
    def __init__(self, model_path: str):
        self.sp     = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def __call__(self, texts, padding: bool = True,
                 truncation: bool = True, max_length: int = 160,
                 return_tensors: Optional[str] = None) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        encoded = []
        for text in texts:
            ids = [self.bos_id] + self.sp.encode(text, out_type=int) + [self.eos_id]
            if truncation and len(ids) > max_length:
                ids = ids[:max_length - 1] + [self.eos_id]
            encoded.append(ids)

        if padding:
            pad_len         = max(len(ids) for ids in encoded)
            attention_masks = []
            padded          = []
            for ids in encoded:
                mask  = [1] * len(ids) + [0] * (pad_len - len(ids))
                ids_p = ids + [self.pad_id] * (pad_len - len(ids))
                padded.append(ids_p)
                attention_masks.append(mask)
            result = {"input_ids": padded, "attention_mask": attention_masks}
        else:
            result = {"input_ids": encoded}

        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
        return result


def build_tokenizer(cfg: CFG):
    """Returns the appropriate tokenizer."""
    if cfg.tokenizer == "sentencepiece":
        return SentencePieceTokenizer(cfg.spm_model_path)
    elif cfg.tokenizer == "bert":
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise ValueError(f"Unknown tokenizer '{cfg.tokenizer}'")

