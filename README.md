# PSG-Free Multi-View Facial Imaging and Attention-Based Fusion for OSA Severity Classification

This is the official implementation of the paper:  
**"PSG-Free Multi-View Facial Imaging and Attention-Based Fusion for OSA Severity Classification"** Submitted to *Expert Systems with Applications (ESWA)*.

---

## 📌 Overview
This repository provides the code for a two-step training strategy and an attention-based fusion model to classify Obstructive Sleep Apnea (OSA) severity using multi-view facial images. Our approach eliminates the need for Polysomnography (PSG) by leveraging deep learning architectures (ResNet, EfficientNet, ViT, Swin) to identify severe OSA (AHI ≥ 30).

## 🚀 Key Features
* **Multi-View Fusion:** Integration of Frontal, Lateral, and Neck views.
* **Attention-Based Fusion:** Adaptive weighting of features from different views for enhanced classification.
* **Reproducibility:** Controlled random seeds for both batch shuffling and data augmentation (spatial & spectral consistency).

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/dongyyyyy/Image-based-OSA-Classification.git
cd Image-based-OSA-Classification

```

```bash
# Run single-view training 
sh shell_files/train/benchmark/train_single_view_image.sh
```
* If you want to change the model architecture and hyper-parameters, please change the shell file!

```bash
# Run multi-view training
sh shell_files/train/benchmark/train_attention_mutli_view_image_three.sh
```
* In the shell file, you have to insert "model weight path" to load pretrained model weight parameter that is trained using 'single-view' image.

## 📂 Project Structure
```text
.
├── config/            # config files directory (argparse)
├── model/             # Backbone architectures (ResNet, EfficientNet, ViT, Swin, multi-view attention)
├── utils/             # Data augmentation, dataloader and utility functions
├── shell_files        # shell files to train the model
├── train              # Train the model
└── test               # Verify the model
```
```
@article{dongyyyyy2024psgfree,
  title={PSG-Free Multi-View Facial Imaging and Attention-Based Fusion for OSA Severity Classification},
  author={Your Name and Co-authors},
  journal={},
  year={}
}
```
