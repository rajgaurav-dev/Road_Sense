# Road Sense Project

Road Sense is an end-to-end object detection project based on the BDD100K Dataset.  
This repository contains solutions for two major parts of the assignment:

- Data Analysis (10 Points) – Completed End-to-End  
- Model Building & Training (5 + 5 Points)

The project follows clean coding practices (PEP8), proper documentation, modular structure, and containerized execution for reproducibility.

---

## Repository Structure

Road_Sense/

Data_Analysis/  
- README.md  
- src/  
- docker/  
- dashboard/  
- reports/  

Model/  
- README.md  
- build_model.py  
- fine_tune.py  
- inference.py  
- configs/  
- notebooks/  

README.md (this file)

Each folder contains its own detailed README explaining implementation details.

---

## 1. Data Analysis (Completed End-to-End)

Folder: `Data_Analysis/`  
Detailed documentation available inside that folder.

### Dataset Used

- BDD100K Object Detection Dataset  
- 100K Images (5.3GB)  
- Labels (107MB)  
- Only 10 detection classes with bounding boxes used  
- Drivable areas / lane marking segmentation NOT used  

### Analysis Performed

#### Dataset Parsing
- Custom parser built to read images and JSON annotations  
- Designed proper data structures for efficient analysis  

#### Class Distribution Analysis
- Distribution of objects across 10 detection classes  
- Object density per image  
- Occlusion percentage  
- Truncation statistics  
- Weather and scene condition distribution  

#### Train vs Validation Split
- Compared distribution consistency  
- Checked class imbalance across splits  

#### Anomaly & Pattern Detection
Examples:
- High occlusion in car class  
- Foggy weather underrepresented  
- Night scenes significant for some classes  
- Certain rare object co-occurrences  

#### Dashboard Visualization
Statistical dashboard created including:
- Class distribution  
- Occlusion rates  
- Weather breakdown  
- Density histograms  
- Unique sample visualization  

#### Interesting Sample Identification
- Highly occluded samples  
- High object density images  
- Rare class combinations  

### Dockerized Environment

The entire data analysis pipeline is containerized.

No additional installation required.  
Fully reproducible.

Run:

```bash
docker build -t road_sense_analysis .
docker run road_sense_analysis 

```


## 2. Model Building & Training

**Folder:** `Model/`

This stage focuses on building and fine-tuning object detection models using MMDetection.

---

### Model Choice

MMDetection framework was used because:

- Highly modular  
- Supports multiple state-of-the-art detectors  
- Large model zoo  
- Strong community support  
- Config-driven architecture  

Models are built using official BDD-compatible config files.

---

### Implemented Scripts

#### Model Builder Script

- Builds model from MMDetection config  
- Loads BDD dataset config  
- Supports:
  - Single image inference  
  - Folder inference  
  - Complete dataset inference  

#### Fine-Tuning Script

- Works with any MMDetection config  
- Can fine-tune any model architecture  
- Config-driven training  
- Flexible training pipeline  

---

### Training Limitation

Training was attempted locally but:

- 5 epochs took approximately 65 hours  

**System specification:**

- Machine Type: 64-bit CPU Machine  
- No dedicated GPU  

Due to hardware limitations, full training was not feasible.

#### Proof of Training Attempt

Paste screenshot here:

---

### Attempt with Google Colab

Training was attempted using Google Colab.

**Issue faced:**

Latest Colab Torch versions are incompatible with the MMDetection version used.

Torch versions available on Colab:

- torch >= 2.x  

This caused:

- CUDA errors  
- Dependency conflicts  
- Version mismatch failures  

As a result, remote training was not successful.

---

## Issues Faced During Assignment

### Pretrained Weight Link Not Working

Some official model zoo weight links were not accessible.

Paste screenshot here:

### Training Time Constraint

CPU-only training was extremely slow (65 hours for 5 epochs).

### Colab Version Compatibility

Torch version mismatch with MMDetection.

---

## What Was Successfully Achieved

- Full Data Analysis with insights  
- Dockerized reproducible analysis  
- Model building pipeline implemented  
- Inference pipeline created  
- Fine-tuning script built  
- Config-driven architecture  
- Clean and modular code  
- PEP8 compliant  
- Proper documentation  

---

## Coding Standards

- Followed PEP8  
- Used:
  - black  
  - pylint  
- Proper docstrings for classes and functions  
- Modular code structure  

---

## How to Run

### Data Analysis

```bash
cd Data_Analysis
docker build -t road_sense_analysis .
docker run road_sense_analysis

```

## Model Inference

```bash
cd Model
python inference.py --config configs/model_config.py --image path/to/image.jpg

```
## Author

Raj Gaurav Tiwari 

---

## Final Note

This repository demonstrates:

- Strong understanding of dataset analysis  
- Ability to derive insights from large-scale datasets  
- Experience working with modern detection frameworks  
- Config-driven deep learning pipelines  
- Containerization skills  
- Practical problem solving under hardware constraints  
