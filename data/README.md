# Data Directory

- Training Datasets:
- [COCO Image Captioning Dataset](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption) + [Validation 2014](http://images.cocodataset.org/zips/val2014.zip)
- [Flickr](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
- [TextCaps](https://www.kaggle.com/datasets/hongtrung/textvqa-textcaps-dataset)

Place them under a single data_directory:
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
  


