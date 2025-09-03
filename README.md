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
2. **Install Jupyter Notebook (if not already installed)**:
```
pip install notebook
```
3. **Run the Jupyter Notebook**:
```
jupyter notebook DINOv2-Linear-Probing-CIFAR10.ipynb

```
  - This will open the Jupyter Notebook in your browser.
  - Run all the cells in the notebook to execute the training process.
4. **Model Weights and Results**:
    
  - The trained model weights are saved as ```linear_head.pth.```

  - The training and testing metrics are saved ```in metrics.json.```

## Results

  - Training Accuracy: ~97.5% after 5 epochs.

  - Test Accuracy: ~95% after 5 epochs.

## Example Predictions

Train Set Predictions:

<img width="1184" height="362" alt="截屏2025-09-03 下午9 20 31" src="https://github.com/user-attachments/assets/e30318ea-9185-4d1e-9cf7-145d18b9ebf1" />

Test Set Predictions:

<img width="1023" height="314" alt="截屏2025-09-03 下午10 05 48" src="https://github.com/user-attachments/assets/2f6a36a4-28f2-4743-9ddc-040bff9c4d95" />

## Conclusion

The project demonstrates the potential of self-supervised learning and transfer learning, where pre-trained models can be adapted for new tasks with minimal labeled data.

## Future Work

  - Fine-tune the entire DINOv2 model for better performance.
  - Experiment with data augmentation techniques to improve generalization.

## References

  - DINOv2: facebook/dinov2-small. "Vision Transformer (small-sized model) trained using DINOv2" https://huggingface.co/facebook/dinov2-small
  - CIFAR-10 Dataset: Alex Krizhevsky, Geoffrey Hinton. "Learning Multiple Layers of Features from Tiny Images." https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com
  - Hugging Face Transformers: Hugging Face. "Transformers Library." https://huggingface.co/docs/transformers/index 

