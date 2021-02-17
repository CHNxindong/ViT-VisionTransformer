# ViT-VisionTransformer

### 0. Overview

This repository contains a simple **unofficial** implementation of **[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)** using PyTorch. 

**ViT (Vision Transformer)** is trained using CIFAR10 and CIFAR100 dataset and its performance is compared with **ResNet34** and **EfficientNet B0**.

### 1. Accuracy

#### a) CIFAR10
| Model | Top 1 Accuracy (%) | Top 5 Accuracy (%) |
|:---:|:---:|:---:|
| ViT | 59.30 | 95.36 |
| ResNet34 | 81.45 | 98.69 |
| EfficientNet B0 | 76.90 | 98.13 |

Low accuracy should not matter, as the original paper stated that they trained the model using ImageNet initially and fine-tuned the model on the CIFAR10 dataset. Here, the model is trained using CIFAR10 from scratch, thus the accuracy should be lower than the original paper.

### 2. Loss and Accuracy during Training

#### a) CIFAR10
| Model | Accuracy | Loss |
|:---:|:---:|:---:|
| ViT | <img src = './results/plots/VisionTransformer Accuracy on CIFAR 10 Dataset.png'> | <img src = './results/plots/VisionTransformer Loss on CIFAR 10 Dataset.png'> |
| ResNet34 | <img src = './results/plots/ResNet Accuracy on CIFAR 10 Dataset.png'> | <img src = './results/plots/ResNet Loss on CIFAR 10 Dataset.png'> |
| EfficientNet B0 | <img src = './results/plots/EfficientNet Accuracy on CIFAR 10 Dataset.png'> | <img src = './results/plots/EfficientNet Loss on CIFAR 10 Dataset.png'> |


### 3. Run the Codes
The default is set to train *ViT*. It will automatically plot the loss and accuracy during train and yield results. If you want to train *ResNet*, 

#### 1) Train 
```
python main.py --model 'resnet'
```

#### 2) Test
```
python main.py --model 'resnet' --phase 'test'
```

To handle more arguments, please refer to `main.py`


### Development Environment
```
- Ubuntu 18.04 LTS
- NVIDIA GFORCE RTX 3090
- CUDA 10.2
- torch 1.6.0
- torchvision 0.7.0
- etc
```
