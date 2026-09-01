# ATLAS: Learning Modality-Specific Coordinate Systems for Shared Representation Learning

ATLAS is a lightweight, from-scratch multimodal representation learning framework that introduces a token-level embedding mechanism to guide modality-specific inputs into a shared representation space. Rather than relying on separate encoders per modality or large pretrained backbones, ATLAS assigns each modality a learned latent basis and uses cross-attention against this modality-specific reference structure while keeping the subsequent encoder and final comparison space fully shared. The approach is evaluated on 8 downstream datasets with only ~20M trainable parameters, isolating the contribution of the embedding mechanism itself from the confound of large-scale pretraining.
Note: MABE in the code refers to modality-aware Basis Embedder.

<img width="1073" height="528" alt="image" src="https://github.com/user-attachments/assets/3f130de0-dff8-4046-9c1e-e47ef81b8be3" />

---
## Data Setup

Training uses three image–caption datasets. Place them under a single data_directory:
```
data/
├── coco-image-caption/
│   ├── annotations_trainval2014/annotations/
│   │   ├── captions_train2014.json
│   │   └── captions_val2014.json
│   ├── train2014/train2014/
│   └── val2014/
├── flickr30k_images/
│   ├── results.csv
│   └── flickr30k_images/
└── TextCaps/
    ├── textcaps_train.csv
    ├── textcaps_val.csv
    └── images_zip/images/train_val/
```
## Repository Structure
```
ATLAS_project/
│
├── data/                   # datasets 
│   └── README.md           # Notes on dataset sources.
│
├── Experimentation/ 
|   ├── PCA/                  # PCA Analysis
│   ├── ImageClassification/  # Image Classification Results
│   └── TextClassification/   # Text Classification Results
│
├── src/                    # Python modules for data processing, modeling, and utilities
│   ├── dataset/
│   │   ├── dataset.py     # Data loading 
│   │   └── tokenizer.py   # Tokenization Process
│   │
│   ├── models/
│   │   ├── MadalityAwareBasisEmbedder.py        # Embedder
│   │   ├── train_valid.py        # Train Valid script
│   │   └── model.py     # Our Architecture
│   │
│   └── utils/
│      └── metrics.py      # Helper functions (plotting, metrics, etc.) 
│
├── results/                # generated figures, metrics, and checkpoints
│   ├── figures/
│   └── checkpoints/
│
├── requirements.txt        # Python dependencies
├──config.py                #configs
├── main.py                # run code
├── .gitignore
├── README.md
└── LICENSE
```
---

## Setup
1. Create and activate a new environment
```bash
conda create -n ATLAS python=3.10 -y
conda activate ATLAS
```
2. Install PyTorch first via conda (matches your CUDA 12.1 pin)
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
3. Install the rest from your requirements.txt
```bash
pip install -r requirements.txt --ignore-installed torch torchvision
```
---
## Training
ATLAS (single GPU)
```bash
python main.py \
  --data_directory /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
  --embedder learnable \
  --tokenizer sentencepiece \
  --pooling_type attn \
  --num_latents 64 \
  --d_model 512 \
  --num_layers 4 \
  --num_head 4 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --seed 42
```
ATLAS (multi-GPU, DDP)
```bash
torchrun --nproc_per_node=2 main.py \
  --data_directory /path/to/data \
  --checkpoint_dir /path/to/checkpoints \
```
---
## Evaluation
1. linear classification(image)
```bash
cd .\Experimentation\ImageClassification\LinearClassification
python ATLAS_classification.py \
  --model_path /path/to/checkpoint.pt \
  --dataset cifar10 \ # can be caltech101, cifar10, cifar100, oxfordpets, tinyimaganet200 or all
  --head linear \ #or mlp
  --epochs 100
```
2. linear classification (text)

```bash
cd .\Experimentation\TextClassification\LinearClassification
python ATLAS_classification.py \
  --model_path /path/to/checkpoint.pt \
  --dataset emotion \ # can be trec or sst2 or all
  --head linear \
  --epochs 100
```
---
## Design Principles
- No pretrained backbones in the main experiments.
- ~20M parameters 
- Shared encoder and comparison space
- Controlled comparison
---
## Citing
@inproceedings{Radford2021LearningTV, title={Learning Transferable Visual Models From Natural Language Supervision}, author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever}, booktitle={ICML}, year={2021} } 

@article{faye2025oneencoder, title={OneEncoder: a lightweight framework for efficient multimodal training}, author={Faye, Bilal and Azzag, Hanane and Lebbah, Mustapha and Bouchaffra, Djamel}, journal={Neural Computing and Applications}, volume={37}, number={22}, pages={18067--18090}, year={2025}, publisher={Springer} }
