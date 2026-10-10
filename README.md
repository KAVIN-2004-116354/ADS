# Automated Optical Inspection (AOI) via Unsupervised Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_13.2-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8)
![Status](https://img.shields.io/badge/Status-Deployment_Ready-success)

## 📌 Project Overview
This project implements a high-speed Automated Optical Inspection (AOI) system for industrial manufacturing. Traditional supervised deep learning models fail in factory environments due to severe data imbalance (defects are rare, making it impossible to collect thousands of defective samples). 

This system solves that by utilizing **unsupervised anomaly detection**. By training exclusively on defect-free ("good") components, the model establishes a mathematical baseline of standard geometry and textures. During real-time inference, it flags any spatial or textural deviations as defects, generating pixel-precise heatmaps for root-cause analysis.

## 🎯 Key Features
* **Unsupervised Learning (PatchCore):** Achieved 100% defect capture rate on the MVTec AD test set without ever being trained on a defective image.
* **Transfer Learning:** Leveraged a pre-trained ResNet-18 backbone as a frozen feature extractor to build a memory bank of normal visual features.
* **Pixel-Precise Localization:** Generates thermal heatmaps overlaying the exact anomaly location, aiding quality control engineers in diagnosing machinery faults.
* **Edge Deployment Simulation:** Optimized PyTorch checkpoint into a deployment-ready `.pt` format, simulating a high-speed conveyor belt with millisecond latency sorting logic.

## 🛠️ Technology Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Computer Vision:** OpenCV, Matplotlib
* **Industrial AI:** Intel `anomalib`
* **Hardware Acceleration:** NVIDIA CUDA 13.2, TensorRT optimization

## 📂 Project Structure
* `download_data.py` - Automated ingestion pipeline for the MVTec AD dataset.
* `train_model.py` - Extracts features from normal images and builds the Coreset subsampled memory bank.
* `export_model.py` - Converts the PyTorch Lightning `.ckpt` into a lightweight inference model.
* `infer_single_image.py` - Runs prediction on a single frame and outputs comparative heatmaps.
* `conveyor_simulation.py` - Simulates a real-time production line, measuring inference latency and dynamically sorting output into `PASS/` and `REJECT/` directories.

## 📊 Results & Performance
Evaluated on the MVTec AD "Bottle" dataset category:
* **Total Test Images:** 83
* **True Positives (Defective parts caught):** 63 / 63
* **True Negatives (Good parts passed):** 20 / 20
* **Latency:** Real-time throughput achieved on an NVIDIA RTX 5050, comfortably supporting industrial conveyor belt line speeds.

## 🚀 How to Run
1. Clone the repository and activate your virtual environment.
2. Install dependencies: `pip install torch torchvision anomalib opencv-python matplotlib`
3. Download the dataset: `python3 download_data.py`
4. Train the PatchCore model: `python3 train_model.py`
5. Run the conveyor simulation: `python3 conveyor_simulation.py`

## 👨‍💻 Author
**Kavin Surya **  
Data Scientist | Machine Learning & Computer Vision
