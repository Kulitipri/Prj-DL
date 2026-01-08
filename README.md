# Malware-as-Image Project

## Overview
The rapid evolution of malicious software (malware) presents a critical challenge to cybersecurity.  
Traditional detection methods, which largely rely on signature-based approaches, are becoming increasingly ineffective against modern evasion techniques such as obfuscation, polymorphism, and packing.  
When malware authors modify a few lines of code, the binary signature changes, allowing the malware to bypass standard antivirus filters.

To address this limitation, this project adopts a **"Malware-as-Image"** approach.  
By visualizing binary executables as grayscale images, we can observe that variants of the same malware family exhibit similar visual textures and structures.  
This insight allows us to leverage powerful Deep Learning techniques typically used in Computer Vision.

## Objectives
The project aims to solve two core problems:

1. **Effective Classification**  
   - Can we accurately classify malware families using visual features (via **VGG16**, **MobileNetV2**, **Custom CNN**)  
   - And sequential structural patterns (via **LSTM**)?

2. **Zero-Day Detection**  
   - Can we detect unknown/new malware anomalies that the system has never seen before (via **Autoencoder**)?

## Repository Structure
This repository is organized into separate branches, each corresponding to a specific algorithm:

- **Branch `vgg16`** → Implementation using VGG16  
- **Branch `mobilenetv2`** → Implementation using MobileNetV2  
- **Branch `custom-cnn`** → Implementation using Custom CNN  
- **Branch `lstm-autoencoder`** → Implementation using LSTM & Autoencoder  

👉 To explore a specific algorithm, please switch to the corresponding branch.

---

## How to Navigate
- Start here in the **main branch** to read the project overview.  
- Switch to the branch of interest to view the code and experiments for that algorithm.  
