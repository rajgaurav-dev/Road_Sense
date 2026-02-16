# BDD100K Detection Dataset Analysis

## Class Distribution (Object-Level)

- Car heavily dominates (102,506 instances).
- Traffic sign and traffic light are moderately represented.
- Person appears less frequently.
- Severe long-tail imbalance for train (15), motor (452), rider (649).

## Image Distribution

- Car appears in almost all images (9,879).
- Rare classes appear in very few images.
- Dataset is strongly car-centric and urban-focused.

---

## Key Dataset Characteristics

- Severe class imbalance (head vs tail classes).
- Multi-scale objects (small traffic signs, medium cars, large buses).
- Dense scenes with overlapping objects.

---

## Impact on Model Design

- Model may bias toward car class without imbalance handling.
- Rare classes may be under-trained.
- Multi-scale feature extraction is essential.

---

## Recommended Loss Functions

- **Classification:** Focal Loss (to handle long-tail imbalance).
- **Bounding Box Regression:** CIoU Loss (better for overlap and dense scenes).
- **Objectness (if applicable):** Binary Cross Entropy.

---

## Forward Pass Impact

In one forward pass:
- Backbone extracts multi-scale features.
- Detection head predicts class scores, bounding boxes, and objectness.
- Focal Loss reduces dominance of easy car samples.
- CIoU improves localization in crowded scenes.

---

## Recommended Architecture

- **Balanced (Speed + Accuracy):** YOLOv8 with FPN and multi-scale training.
- **High Accuracy Priority:** Cascade R-CNN with FPN.

Both architectures handle dense scenes and multi-scale objects effectively.

# Class: car

- Present in 69,072 images with 713,211 total instances.
- High object density with an average of 10.33 cars per image.
- Strong occlusion characteristic (67.74% occluded).
- Low truncation rate (9.33%), indicating most objects are fully within frame.
- Predominantly appears in clear weather conditions.
- Foggy conditions are severely underrepresented.
- Significant representation in night scenes.
- Traffic light color attribute is mostly irrelevant for this class.

## Impact on Model Selection

Since most car instances are heavily occluded, the model must learn to detect partially visible objects and overlapping bounding boxes. The high object density increases scene complexity and makes detection more challenging.

Although the dataset is dominated by crowded scenes, it is important that the model generalizes well to scenarios with a single car. A strong feature representation will help ensure robustness across both dense and sparse scenes.

## Preferred Architecture

A two-stage detector is preferred because region proposal networks help isolate candidate object regions, which improves performance in occluded and crowded environments.

If accuracy is prioritized over speed, a two-stage model such as Faster R-CNN or Cascade R-CNN with Feature Pyramid Network (FPN) is suitable.

A multi-scale feature representation is essential to detect cars of different sizes, ensuring that both small and large vehicles are properly captured in dense traffic scenarios.
