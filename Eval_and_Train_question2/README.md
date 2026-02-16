# BDD100K Object Detection – MMDetection 3.x Implementation

This repository contains a custom implementation of the **BDD100K Object Detection pipeline** adapted for **MMDetection 3.x**.

It includes:

- Custom dataset registration
- BDD100K → COCO format conversion
- Training and inference scripts
- Configs compatible with MMDet 3.x

---

## 📁 Folder Description

### 1️⃣ configs/

Contains model and dataset configuration files.

- Converted and adapted for **MMDetection 3.x**
- Similar structure to official BDD configs
- Includes:
  - Model architecture
  - Dataset configuration
  - Training pipeline
  - Evaluation settings

These configs are fully compatible with MMEngine + MMDetection 3.x.

---

### 2️⃣ Dataset/

Contains the custom **BDD100K Dataset class**.

- Implements BDD100K detection dataset
- Registered inside **MMDetection Dataset Registry**
- Ensures compatibility with MMDet training & evaluation pipelines
- Handles dataset loading and annotation parsing

Make sure the dataset is properly registered before training.

---

### 3️⃣ utils/

Contains utility functions.

Main functionality:

- Convert **BDD100K annotation format → COCO format**
- MMDetection pipelines require COCO-style annotations
- Ensures compatibility with:
  - Training
  - Validation
  - Testing

---

## 🚀 train.py

Training script for the model.

### Usage

```bash
python train.py

```
## ⚠️ Important

Before running training:

1. Open `bdd100k.py`
2. Update:
   - Image folder path
   - JSON annotation file path

Make sure dataset paths are correctly set.

---

## 🔍 inference.py

Inference script.

### Supports:

- Single image inference  
- Folder image inference  
- Test dataset inference  

### Usage

```bash
python inference.py --config configs/your_config.py --weights path/to/weights.pth --input path/to/image_or_folder
```

### Notes

- If `--weights` is not provided:
  - Model will run with **random weights**
  - Output will not be meaningful (only for testing pipeline)

---

## Dataset Format

This project uses:

- Original BDD100K annotations
- Converted to COCO format
- Compatible with MMDetection 3.x

---

## Workflow Summary

1. Convert BDD100K → COCO format (`utils`)
2. Register Dataset class (`Dataset` folder)
3. Update paths in `configs\datasets\bdd100k.py`
4. Train using `train.py`
5. Run inference using `inference.py`

---

## Customization

To use a different dataset:

- Modify conversion utility
- Update Dataset class
- Update config file
