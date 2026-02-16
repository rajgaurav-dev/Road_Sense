"""
Dataset configuration for BDD100K object detection using MMDetection 3.x.

This configuration defines:

- Dataset type and root directory
- Image normalization parameters
- Training and validation pipelines
- DataLoader configurations
- COCO-style evaluation settings




"""

dataset_type = "BDD100KDetDataset"
data_root = "C:/Bosch_assignment/Dataset"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

"""
Training data preprocessing pipeline.

Steps:
1. Load image from file.
2. Load bounding box annotations.
3. Resize image while keeping aspect ratio.
4. Apply random horizontal flip augmentation.
5. Normalize using ImageNet statistics.
6. Pad image to be divisible by 32.
7. Pack inputs into MMDetection format.
"""
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="PackDetInputs"),
]

"""
Validation and test preprocessing pipeline.

This pipeline does not include augmentation.
It ensures deterministic preprocessing during evaluation.
"""
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="PackDetInputs"),
]

"""
Training DataLoader configuration.

- Batch size: 4
- Uses DefaultSampler with shuffling
- Persistent workers enabled for faster loading
"""
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=(
            data_root + "/bdd100k_labels_release/bdd100k/labels/"
            "bdd_val_to_coco_val.json"
        ),
        data_prefix=dict(
            img=(data_root + "/bdd100k_images_100k/bdd100k/images/100k/val")
        ),
        pipeline=train_pipeline,
    ),
)

"""
Validation DataLoader configuration.

- Batch size: 1
- No shuffling
- Uses COCO-format annotations
- test_mode=True ensures evaluation behavior
"""
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "/jsons/det_val_cocofmt.json",
        data_prefix=dict(img=data_root + "/images/100k/val"),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

"""
Test DataLoader configuration.

Reuses validation DataLoader settings.
"""
test_dataloader = val_dataloader

"""
Evaluation configuration using COCO metrics.

Metric:
- Bounding box detection (mAP)
"""
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "/jsons/det_val_cocofmt.json",
    metric="bbox",
)

test_evaluator = val_evaluator
