# YOLO Custom Dataset Training Tutorial

A step-by-step tutorial for training YOLO (You Only Look Once) object detection models on custom datasets using Google Colab. This guide walks you through setting up a head detection project, as an example, from dataset acquisition to model training.

## 📋 Overview

This tutorial demonstrates how to:
- Set up Kaggle API for dataset access
- Download and prepare a custom dataset (Head Detection CCTV)
- Configure YAML files for YOLO training
- Train YOLOv8 models on Google Colab with GPU acceleration
- Handle common dataset structure issues

## 🚀 Quick Start

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zsW6OcoMUE38k6t-UhBrERCw6TY7JTKx?usp=sharing)

1. Click the "Open in Colab" button above
2. Upload your `kaggle.json` file when prompted
3. Run all cells sequentially
4. Monitor training progress and results

## 🛠️ Prerequisites

- Google Colab account
- Kaggle account with API key
- Basic understanding of Python and object detection

## 📥 Installation & Setup

### 1. Kaggle API Setup
- Visit [Kaggle](https://www.kaggle.com/) → Account → API → Create New API Token
- Download `kaggle.json` file
- Upload the file when prompted in the Colab notebook

### 2. Environment Setup
The notebook automatically installs required packages:
```python
pip install kaggle ultralytics
```

### 🎯 Dataset Information
This tutorial uses the Head Detection CCTV Dataset from Kaggle, containing:
- Training images and labels
- Validation images and labels
- Single class: "head"

### ⚙️ Configuration
#### Dataset YAML File
The ```python head_dataset.yaml``` file configures:
- Training and validation image paths
- Number of classes (1 - "head")
- Class names

#### Training Parameters
- Model: YOLOv8s (small)
- Image size: 640x640
- Epochs: 50
- Batch size: 16
- Workers: 2

### 📊 Training
Run the training cell to start the process:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="head_dataset.yaml",
    imgsz=640,
    epochs=50,
    batch=16,
    workers=2,
)
```

### 📈 Results
After training, you can expect:
- Training/validation data
- Model weights saved in runs/detect/train/
- Performance metrics (mAP, precision, recall)

### 🐛 Common Issues & Solutions
#### Folder Name Mismatch
The notebook automatically handles the "valid" to "val" folder renaming.

### Kaggle API Errors
- Ensure ```python kaggle.json``` is properly uploaded
- Check file permissions: ```python chmod 600 /root/.kaggle/kaggle.json```

### GPU Not Available
In Colab: Runtime → Change runtime type → GPU

### 🎓 Educational Purpose
This tutorial is designed for educational use in computer vision and machine learning courses. It demonstrates practical implementation of object detection pipelines.

### 🤝 Contributing
Feel free to submit issues and enhancement requests!

### 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

#### 🙏 Acknowledgments
- Ultralytics for YOLOv8
- Kaggle for the dataset and platform

### 📞 Support
For questions related to this tutorial, please open an issue in this repository.
