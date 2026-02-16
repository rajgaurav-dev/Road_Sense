# BDD100K Detection Dataset Analysis

This project performs structured analysis of the BDD100K object detection dataset.  
It includes dataset parsing, statistical analysis, class-wise deep analysis, visualization tools, and model design recommendations based on observed data patterns.

---

## Project Structure

bddquestion1/
│
├── src/
│ ├── bddparser.py
│ ├── classwise_analyzer.py
│ ├── plotter.py
│ ├── dashboard.py
│
├── analysis.md
├── statistics.md
└── README.md


All core implementation files are located inside the `src/` directory.

---

## 1. Dataset Parsing

File: `src/bddparser.py`

The `BDDParser` class:

- Loads BDD100K JSON files (train/val).
- Extracts image-level attributes (weather, time of day).
- Extracts object-level annotations:
  - Class
  - Bounding box
  - Area
  - Occlusion
  - Truncation
- Computes:
  - Object-level class distribution
  - Image-level class distribution

The internal data structure maps:


All core implementation files are located inside the `src/` directory.

---

## 1. Dataset Parsing

File: `src/bddparser.py`

The `BDDParser` class:

- Loads BDD100K JSON files (train/val).
- Extracts image-level attributes (weather, time of day).
- Extracts object-level annotations:
  - Class
  - Bounding box
  - Area
  - Occlusion
  - Truncation
- Computes:
  - Object-level class distribution
  - Image-level class distribution

The internal data structure maps:


All core implementation files are located inside the `src/` directory.

---

## 1. Dataset Parsing

File: `src/bddparser.py`

The `BDDParser` class:

- Loads BDD100K JSON files (train/val).
- Extracts image-level attributes (weather, time of day).
- Extracts object-level annotations:
  - Class
  - Bounding box
  - Area
  - Occlusion
  - Truncation
- Computes:
  - Object-level class distribution
  - Image-level class distribution

The internal data structure maps:

image_name → {
attributes,
annotations
}


This allows efficient statistical and class-wise analysis.

---

## 2. Dataset Analysis

Detailed analysis is documented in:

- `analysis.md`
- `statistics.md`

### Key Findings

- Severe long-tail class imbalance.
- Car class heavily dominates.
- Rare classes (e.g., train) are extremely underrepresented.
- Dataset is urban and vehicle-centric.
- Multi-scale objects are present (small signs, medium cars, large buses).
- Dense scenes with significant overlap and occlusion.

Train and validation splits show consistent distribution patterns with no major drift.

---

## 3. Class-Wise Deep Analysis

File: `src/classwise_analyzer.py`

Performs detailed class-level analysis including:

- Number of images containing a class
- Total object instances
- Average objects per image
- Occlusion percentage
- Truncation percentage
- Weather distribution
- Time-of-day distribution

Example:

python src/classwise_analyzer.py --json path/to/train.json --class_name car


---

## 4. Visualization

### CLI Plotting

File: `src/plotter.py`

Generates:

- Object-level distribution plots
- Image-level distribution plots
- Percentage view
- Log-scale visualization

Example:

python src/plotter.py --json path/to/train.json --stat object


---

### Interactive Dashboard

File: `src/dashboard.py`

Gradio-based dashboard supporting:

- Class filtering
- Weather filtering
- Time-of-day filtering
- Occlusion and truncation filtering
- Object-level statistics
- Image-level statistics
- Filtered image visualization with bounding boxes

Run:

python src/dashboard.py


---

## 5. Model Design Recommendations

Based on dataset characteristics:

### Identified Challenges

- Long-tail class imbalance
- Multi-scale object detection
- Dense overlapping scenes
- High occlusion in dominant classes

### Recommended Loss Functions

- Classification: Focal Loss
- Bounding Box Regression: CIoU Loss
- Objectness (if applicable): Binary Cross Entropy

### Recommended Architectures

Balanced (Speed + Accuracy):
- YOLOv8 with FPN and multi-scale training

Accuracy Priority:
- Faster R-CNN with FPN
- Cascade R-CNN with FPN

---

## 6. Requirements

Install dependencies:

pip install matplotlib gradio opencv-python numpy


Python version: 3.8+

---

## 7. Summary

This project demonstrates:

- Structured dataset parsing
- Statistical and split-level analysis
- Long-tail imbalance identification
- Class-wise deep analysis
- Visualization tooling
- Model and loss selection reasoning

The analysis reflects understanding of:

- Class imbalance handling
- Multi-scale feature extraction
- Occlusion challenges
- Architecture trade-offs in object detection
