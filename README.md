# YOLO adversarial attack

## Overview

The aim of the project is to implement **adversarial attack** against YOLO-based object detectors using Untargeted **FGSM** (Fast Gradient Sign Method). 

Moreover, evalutaion of confusion matrix's will be conducted for the validation dataset before and after attack.

The goal is to generate visually small perturbations that are hardly recognizable for a human and significantly reduce detection accuracy.

## Tech Stack

**Languages & Frameworks**
- **Python** – Core programming language  
- **PyTorch** – Deep learning framework for training and evaluating YOLO models  
- **YOLO (You Only Look Once)** – Object detection architecture (version-specific: YOLOv8)

**Scientific & Utility Libraries**
- **NumPy** – Numerical operations and array manipulation  
- **Pandas** – Data loading and preprocessing  
- **Matplotlib** – Visualization and plotting  

**Deployment & Environment**
- **Docker** (optional) – Containerized environment for reproducibility  
- **Jupyter Notebooks** – Interactive experimentation and prototyping
