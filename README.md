# Road Sense Project

Road Sense is an end-to-end object detection project based on the BDD100K Dataset.  
This repository contains solutions for two major parts of the assignment:

- Data Analysis (10 Points) – Completed End-to-End  
- Model Building & Training (5 + 5 Points)

The project follows clean coding practices (PEP8), proper documentation, modular structure, and containerized execution for reproducibility.

---

## Repository Structure

Road_Sense/

Bdd_Analysis_question1/  
- README.md  
- src/  
- output_jsons/ 
- analysis.md
- statistics.md

Model/  
- README.md  
- configs/ 
- Dataset  
- inference.py  
- finetune.py/  
- utils/  

README.md (this file)

Each folder contains its own detailed README explaining implementation details.

---

## 1. Data Analysis (Completed End-to-End)

Folder: `Data_Analysis/`  
  
Detailed documentation, implementation, dataset insights, and execution instructions are available in:

[Bdd_Analysis_question1/README.md](./Bdd_Analysis_question1/README.md)


## 2. Model Building & Training

**Folder:** `Model/`

This stage focuses on building and fine-tuning object detection models using MMDetection.

---
[Eval_and_Train_question2/README.md](./Eval_and_Train_question2/README.md)


## 3. Training Limitation

Training was attempted locally but:

- 5 epochs took approximately 65 hours  

**System specification:**

- Machine Type: 64-bit CPU Machine  
- No dedicated GPU  

Due to hardware limitations, fine tuning was not feasible.

#### Proof of Training Attempt

![Training Proof](assets/training_log.png)
---

## 4. Attempt with Google Colab

Training was attempted using Google Colab.

**Issue faced:**

Latest Colab Torch versions are incompatible with the MMDetection version used.

This caused:

- CUDA errors  
- Dependency conflicts  
- Version mismatch failures  

As a result, remote training was not successful.

---

## 5. Issues Faced During Assignment

### Pretrained Weight Link Not Working

Some official model zoo weight links were not accessible.

[weight_link](https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_1x_det_bdd100k.pth)

![Error](assets/Error.png)

## 6. Environment Setup


```bash
git clone https://github.com/rajgaurav-dev/Road_Sense.git
cd Road_Sense
docker build -t road_sense_analysis .
docker run road_sense_analysis

```


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


## Author

Raj Gaurav Tiwari 

