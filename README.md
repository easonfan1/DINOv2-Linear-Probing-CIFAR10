# DINOv2 Linear Probing for CIFAR-10 Classification

This project demonstrates the use of the **DINOv2** (Self-Supervised Vision Transformer) model for image classification on the **CIFAR-10** dataset using **linear probing**.

## Objective
The goal of this project is to apply **self-supervised learning** using DINOv2 to perform **image classification** with minimal labeled data. The DINOv2 backbone is used as a feature extractor, and a **linear classifier head** is trained to classify images into one of the 10 CIFAR-10 categories.

## Project Overview
- **Model**: DINOv2 (Self-Supervised Vision Transformer)
- **Dataset**: CIFAR-10 (10 classes)
- **Technique**: Linear Probing (Training a simple linear classifier on top of DINOv2 features)
- **Tools**: Python, PyTorch, Hugging Face Transformers, Matplotlib

## Requirements
To run this project, you need to install the following dependencies:
```
pip install torch torchvision transformers scikit-learn tqdm matplotlib

```
## Usage
1. **Clone this repository**:
```
git clone https://github.com/easonfan1/DINOv2-Linear-Probing-CIFAR10.git
cd DINOv2-Linear-Probing-CIFAR10
```
2. **Run the training script**:
```
