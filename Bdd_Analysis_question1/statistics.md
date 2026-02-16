# BDD100K Detection Dataset Analysis

---

# 1. Checking the Distribution of Training Samples for Object Detection

## Parser and Data Structure

A custom parser (`BDDParser`) was implemented to:

- Load the BDD100K JSON file.
- Store image-level attributes (weather, time of day).
- Extract object-level annotations (class, bbox, occlusion, truncation).
- Compute:
  - Object-level class distribution.
  - Image-level class presence distribution.

The data structure used:

- Dictionary mapping image name → {
    attributes,
    annotations (list of objects with bbox + metadata)
  }

This structure allows efficient class-wise and split-wise analysis.

---

## Training Set Distribution (train.json)

Total Images: 69,863

### Object-Level Distribution

- Car: 713,211
- Traffic sign: 239,686
- Traffic light: 186,117
- Person: 91,349
- Truck: 29,971
- Bus: 11,672
- Bike: 7,210
- Rider: 4,517
- Motor: 3,002
- Train: 136

### Image-Level Distribution

- Car: 69,072 images
- Traffic sign: 57,154
- Traffic light: 39,237
- Person: 22,076
- Truck: 18,890
- Bus: 8,993
- Bike: 4,343
- Rider: 3,586
- Motor: 2,284
- Train: 105

---

## Observations (Train Split)

- Car is overwhelmingly dominant.
- Clear long-tail distribution across classes.
- Train class is extremely underrepresented (136 instances).
- Traffic sign and traffic light are strongly present.
- Dataset is urban and vehicle-centric.

---

# 2. Analysis of Train and Validation Splits

## General Observation

Both splits show:

- Similar class dominance pattern.
- Car as the primary object class.
- Rare classes remain rare in both splits.
- No drastic distribution shift between train and val.

## Split Characteristics

- Train split contains significantly more samples.
- Validation split maintains class imbalance pattern.
- Rare classes (e.g., train) appear very few times in both splits.

## Conclusion

- Train and validation splits are consistent in distribution.
- No major class distribution drift observed.
- However, rare classes may not be sufficiently represented for robust evaluation.

---

# 3. Identified Anomalies and Patterns

## 1. Severe Class Imbalance

- Car class dominates all other classes.
- Train class has extremely low representation.
- Long-tail behavior is evident.

Impact:
- Model may bias toward car.
- Rare classes may have poor recall.

---

## 2. Car-Centric Dataset

- Car appears in almost every image.
- Dataset strongly reflects urban driving scenarios.

Impact:
- Model may generalize well for vehicle detection.
- Less balanced for rare object detection.

---

## 3. Multi-Scale Objects

- Traffic signs and lights are typically small.
- Cars and trucks are medium-sized.
- Buses and trains are large.

Impact:
- Multi-scale feature extraction (FPN) is necessary.

---

## 4. Rare Class Risk

- Train class appears in only 105 training images.
- Insufficient diversity for stable learning.

Impact:
- May require class re-weighting or focal loss.

---

# Overall Conclusion

- Dataset exhibits strong long-tail imbalance.
- Car class dominates both object-level and image-level counts.
- Train and validation splits are distributionally consistent.
- Multi-scale detection architecture is required.
- Loss functions such as Focal Loss should be considered to address class imbalance.
