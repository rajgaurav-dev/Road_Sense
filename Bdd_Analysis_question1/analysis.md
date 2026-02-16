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

## Recommended Loss Functions

- **Classification:** Focal Loss (to handle long-tail imbalance).
- **Bounding Box Regression:** CIoU Loss (better for overlap and dense scenes).
- **Objectness (if applicable):** Binary Cross Entropy.

---

## Impact on Model Selection

Since most car instances are heavily occluded, the model must learn to detect partially visible objects and overlapping bounding boxes. The high object density increases scene complexity and makes detection more challenging.

Although the dataset is dominated by crowded scenes, it is important that the model generalizes well to scenarios with a single car. A strong feature representation will help ensure robustness across both dense and sparse scenes.

---

## Preferred Architecture

A two-stage detector is preferred because region proposal networks help isolate candidate object regions, which improves performance in occluded and crowded environments.

If accuracy is prioritized over speed, a two-stage model such as Faster R-CNN or Cascade R-CNN with Feature Pyramid Network (FPN) is suitable.

A multi-scale feature representation is essential to detect cars of different sizes, ensuring that both small and large vehicles are properly captured in dense traffic scenarios.


# Class: car

- Present in 69,072 images with 713,211 total instances.
- High object density with an average of 10.33 cars per image.
- Strong occlusion characteristic (67.74% occluded).
- Low truncation rate (9.33%), indicating most objects are fully within frame.
- Predominantly appears in clear weather conditions.
- Foggy conditions are severely underrepresented.
- Significant representation in night scenes.
- Traffic light color attribute is mostly irrelevant for this class.

# Class: train

- Present in 105 images with 136 total instances.
- Very low object density with an average of 1.3 trains per image.
- High occlusion characteristic (58.82% occluded).
- High truncation rate (27.94%), indicating trains are often partially visible.
- Predominantly appears in clear weather conditions.
- Limited representation across snowy, rainy, and overcast conditions.
- Mostly present in daytime scenes with fewer night and dawn/dusk samples.
- Traffic light color attribute is largely irrelevant for this class.


