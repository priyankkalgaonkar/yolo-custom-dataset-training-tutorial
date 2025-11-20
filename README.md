# YOLO Custom Dataset Training Tutorial

A comprehensive step-by-step tutorial for training YOLO (You Only Look Once) object detection models on custom datasets using Google Colab. This guide covers everything from dataset preparation to model training and results interpretation for computer vision projects.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration Setup](#configuration-setup)
- [Training](#training)
- [Results Interpretation](#results-interpretation)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Introduction

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. This tutorial will guide you through training YOLOv8 (or YOLOv5) on your custom dataset, enabling you to detect specific objects relevant to your use case.

**What you'll learn:**
- How to prepare and annotate custom datasets for YOLO
- How to configure YOLO training parameters
- How to train YOLO models on Google Colab (free GPU)
- How to evaluate and interpret training results
- How to use your trained model for inference

## Prerequisites

Before starting this tutorial, ensure you have:

### Required Knowledge
- Basic understanding of Python programming
- Familiarity with deep learning concepts (recommended but not mandatory)
- Basic understanding of object detection

### Required Tools
- **Google Account**: For accessing Google Colab
- **Google Drive**: For storing datasets and models (minimum 5GB free space recommended)
- **GPU Access**: Google Colab provides free GPU access (T4 GPU)

### Optional Tools
- **Roboflow Account**: For dataset annotation and management (free tier available)
- **Label Studio** or **LabelImg**: Alternative annotation tools

## Installation

### Step 1: Set Up Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook: `File > New Notebook`
3. Enable GPU runtime:
   - Navigate to `Runtime > Change runtime type`
   - Select `T4 GPU` under Hardware accelerator
   - Click `Save`

### Step 2: Install Required Packages

Run the following commands in your Colab notebook:

```python
# Install Ultralytics (YOLOv8)
!pip install ultralytics

# Verify installation
import ultralytics
ultralytics.checks()

# Import necessary libraries
from ultralytics import YOLO
import torch
import os
from google.colab import drive

# Verify GPU availability
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 3: Mount Google Drive

```python
# Mount Google Drive to save models and datasets
drive.mount('/content/drive')
```

## Dataset Preparation

### Dataset Structure

Organize your dataset following the YOLO format:

```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

### Annotation Format

YOLO uses a specific annotation format for each image. Each `.txt` file contains one row per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Integer representing the object class (0-indexed)
- `x_center, y_center`: Center coordinates of bounding box (normalized 0-1)
- `width, height`: Width and height of bounding box (normalized 0-1)

**Example annotation file (`image1.txt`):**
```
0 0.5 0.5 0.3 0.4
1 0.25 0.75 0.15 0.2
```

### Creating Annotations

#### Option 1: Using Roboflow (Recommended)
1. Sign up at [Roboflow](https://roboflow.com/)
2. Create a new project
3. Upload your images
4. Annotate images using the built-in tool
5. Export in YOLOv8 format
6. Download the dataset

#### Option 2: Using LabelImg
1. Install LabelImg: `pip install labelImg`
2. Run: `labelImg`
3. Open directory with images
4. Select "YOLO" as output format
5. Draw bounding boxes and assign classes
6. Save annotations

#### Option 3: Using Label Studio
1. Install: `pip install label-studio`
2. Run: `label-studio start`
3. Create a project and import images
4. Configure YOLO labeling interface
5. Export in YOLO format

### Dataset Split Recommendations

- **Training Set**: 70-80% of total images
- **Validation Set**: 10-20% of total images
- **Test Set**: 10% of total images (optional but recommended)

**Minimum dataset size recommendations:**
- Small projects: 100-500 images per class
- Medium projects: 500-2000 images per class
- Large projects: 2000+ images per class

## Configuration Setup

### Step 1: Create Dataset YAML File

Create a `data.yaml` file to define your dataset configuration:

```yaml
# data.yaml
path: /content/drive/MyDrive/yolo_dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')
test: test/images    # test images (optional)

# Classes
names:
  0: person
  1: car
  2: bicycle
  # Add more classes as needed

# Number of classes
nc: 3
```

### Step 2: Upload Dataset to Google Drive

```python
# Create directory structure in Google Drive
!mkdir -p /content/drive/MyDrive/yolo_dataset/train/images
!mkdir -p /content/drive/MyDrive/yolo_dataset/train/labels
!mkdir -p /content/drive/MyDrive/yolo_dataset/val/images
!mkdir -p /content/drive/MyDrive/yolo_dataset/val/labels

# Upload your dataset to these directories
# or use wget/gdown to download from cloud storage
```

### Step 3: Configure Training Hyperparameters

Key hyperparameters to consider:

```python
# Training configuration
config = {
    'epochs': 100,           # Number of training epochs
    'batch': 16,             # Batch size (adjust based on GPU memory)
    'imgsz': 640,            # Image size (640, 1280, etc.)
    'patience': 50,          # Early stopping patience
    'save': True,            # Save checkpoints
    'device': 0,             # GPU device (0 for cuda:0)
    'workers': 8,            # Number of worker threads
    'project': 'yolo_runs',  # Project name
    'name': 'exp',           # Experiment name
    'exist_ok': True,        # Overwrite existing project
    'pretrained': True,      # Use pretrained weights
    'optimizer': 'auto',     # Optimizer (auto, SGD, Adam, AdamW)
    'verbose': True,         # Verbose output
    'seed': 0,               # Random seed for reproducibility
    'deterministic': True,   # Deterministic mode
    'single_cls': False,     # Treat as single-class dataset
    'rect': False,           # Rectangular training
    'cos_lr': False,         # Cosine learning rate scheduler
    'close_mosaic': 10,      # Disable mosaic augmentation for final epochs
    'resume': False,         # Resume from last checkpoint
    'amp': True,             # Automatic Mixed Precision training
    'fraction': 1.0,         # Dataset fraction to train on
    'profile': False,        # Profile ONNX and TensorRT speeds
    'freeze': None,          # Freeze layers (None, int, or list)
    'lr0': 0.01,             # Initial learning rate
    'lrf': 0.01,             # Final learning rate factor
    'momentum': 0.937,       # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # Optimizer weight decay
    'warmup_epochs': 3.0,    # Warmup epochs
    'warmup_momentum': 0.8,  # Warmup initial momentum
    'warmup_bias_lr': 0.1,   # Warmup initial bias learning rate
    'box': 7.5,              # Box loss gain
    'cls': 0.5,              # Class loss gain
    'dfl': 1.5,              # DFL loss gain
    'pose': 12.0,            # Pose loss gain (pose models only)
    'kobj': 1.0,             # Keypoint obj loss gain (pose models only)
    'label_smoothing': 0.0,  # Label smoothing epsilon
    'nbs': 64,               # Nominal batch size
    'hsv_h': 0.015,          # HSV-Hue augmentation
    'hsv_s': 0.7,            # HSV-Saturation augmentation
    'hsv_v': 0.4,            # HSV-Value augmentation
    'degrees': 0.0,          # Rotation augmentation (degrees)
    'translate': 0.1,        # Translation augmentation (fraction)
    'scale': 0.5,            # Scale augmentation (gain)
    'shear': 0.0,            # Shear augmentation (degrees)
    'perspective': 0.0,      # Perspective augmentation (0-0.001)
    'flipud': 0.0,           # Vertical flip augmentation (probability)
    'fliplr': 0.5,           # Horizontal flip augmentation (probability)
    'mosaic': 1.0,           # Mosaic augmentation (probability)
    'mixup': 0.0,            # Mixup augmentation (probability)
    'copy_paste': 0.0,       # Copy-paste augmentation (probability)
}
```

## Training

### Step 1: Load Pre-trained Model

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
# Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), 
#          yolov8l.pt (large), yolov8x.pt (xlarge)
model = YOLO('yolov8n.pt')  # Start with nano for faster training
```

### Step 2: Start Training

```python
# Train the model
results = model.train(
    data='/content/drive/MyDrive/yolo_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_yolo',
    project='/content/drive/MyDrive/yolo_runs',
    device=0,
    patience=50,
    save=True,
    plots=True
)
```

### Step 3: Monitor Training

Training progress will display:
- **Loss values**: Box loss, class loss, DFL loss
- **Metrics**: Precision, Recall, mAP50, mAP50-95
- **Learning rate**: Current learning rate
- **GPU memory**: Memory usage

Example output:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100      3.45G      1.234      2.456      1.789         45        640
  2/100      3.45G      1.123      2.234      1.678         42        640
...
```

### Step 4: Resume Training (if interrupted)

```python
# Resume training from last checkpoint
model = YOLO('/content/drive/MyDrive/yolo_runs/custom_yolo/weights/last.pt')
results = model.train(resume=True)
```

### Training Commands Summary

```python
# Basic training
model.train(data='data.yaml', epochs=100)

# Training with custom parameters
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    device=0
)

# Multi-GPU training
model.train(data='data.yaml', epochs=100, device=[0,1])

# Training from scratch (no pre-trained weights)
model = YOLO('yolov8n.yaml')  # Use YAML config instead of .pt file
model.train(data='data.yaml', epochs=300, pretrained=False)
```

## Results Interpretation

### Training Outputs

After training, you'll find these files in your output directory:

```
yolo_runs/custom_yolo/
├── weights/
│   ├── best.pt          # Best model weights
│   └── last.pt          # Last epoch weights
├── results.csv          # Training metrics
├── results.png          # Training curves
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png         # F1 score curve
├── PR_curve.png         # Precision-Recall curve
├── P_curve.png          # Precision curve
├── R_curve.png          # Recall curve
└── labels.jpg           # Label distribution
```

### Key Metrics

#### 1. **Loss Values**
- **Box Loss**: Measures bounding box localization accuracy (lower is better)
- **Class Loss**: Measures classification accuracy (lower is better)
- **DFL Loss**: Distribution Focal Loss for bounding box regression (lower is better)

Target: All losses should decrease and stabilize over epochs.

#### 2. **Precision**
- Percentage of correct positive predictions
- Formula: `TP / (TP + FP)`
- Target: > 0.8 (80%)

#### 3. **Recall**
- Percentage of actual positives correctly identified
- Formula: `TP / (TP + FN)`
- Target: > 0.8 (80%)

#### 4. **mAP (mean Average Precision)**
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
  - Target: > 0.8 for good models
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
  - Target: > 0.5 for good models
  - This is the primary metric for COCO dataset

#### 5. **F1 Score**
- Harmonic mean of Precision and Recall
- Formula: `2 × (Precision × Recall) / (Precision + Recall)`
- Target: > 0.8

### Visualizing Results

```python
# Load and display training results
from IPython.display import Image, display

# Display training curves
display(Image('/content/drive/MyDrive/yolo_runs/custom_yolo/results.png'))

# Display confusion matrix
display(Image('/content/drive/MyDrive/yolo_runs/custom_yolo/confusion_matrix.png'))

# Display Precision-Recall curve
display(Image('/content/drive/MyDrive/yolo_runs/custom_yolo/PR_curve.png'))
```

### Validating the Model

```python
# Validate the trained model
metrics = model.val()

print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

### Making Predictions

```python
# Load the best model
model = YOLO('/content/drive/MyDrive/yolo_runs/custom_yolo/weights/best.pt')

# Predict on a single image
results = model.predict(source='path/to/image.jpg', save=True, conf=0.5)

# Predict on multiple images
results = model.predict(source='path/to/images/', save=True, conf=0.5)

# Predict on video
results = model.predict(source='path/to/video.mp4', save=True, conf=0.5)

# Display results
for result in results:
    result.show()  # Display image with predictions
    print(result.boxes)  # Print bounding boxes
```

### Exporting the Model

```python
# Export to different formats
model.export(format='onnx')        # ONNX format
model.export(format='torchscript') # TorchScript
model.export(format='coreml')      # CoreML (for iOS)
model.export(format='tflite')      # TensorFlow Lite (for mobile)
model.export(format='engine')      # TensorRT (for NVIDIA GPUs)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. **Out of Memory (OOM) Error**
```
CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
- Reduce batch size: `batch=8` or `batch=4`
- Reduce image size: `imgsz=416` instead of `imgsz=640`
- Use a smaller model: `yolov8n.pt` instead of `yolov8l.pt`
- Clear cache: `torch.cuda.empty_cache()`

#### 2. **Low mAP Scores**

**Solutions:**
- Increase training epochs (try 200-300)
- Check annotation quality (verify labels are correct)
- Increase dataset size (more diverse images)
- Adjust confidence threshold for predictions
- Use data augmentation
- Try different model sizes (yolov8m.pt or yolov8l.pt)

#### 3. **Model Not Detecting Objects**

**Solutions:**
- Lower confidence threshold: `conf=0.25` or `conf=0.1`
- Verify data.yaml paths are correct
- Check if class IDs match between training and inference
- Ensure image preprocessing is consistent
- Verify annotation format is correct

#### 4. **Training Not Converging**

**Solutions:**
- Reduce learning rate: `lr0=0.001`
- Increase warmup epochs: `warmup_epochs=5`
- Check for corrupted images or labels
- Verify class distribution is balanced
- Try different optimizer: `optimizer='Adam'`

#### 5. **Dataset Loading Errors**

**Solutions:**
- Verify all paths in data.yaml are correct
- Ensure images and labels have matching names
- Check file permissions
- Verify annotation format (one line per object)
- Remove corrupted images

#### 6. **Google Colab Disconnection**

**Solutions:**
- Save checkpoints regularly
- Use Google Colab Pro for longer sessions
- Run training in shorter sessions
- Use `resume=True` to continue training
- Keep browser tab active

### Debugging Tips

```python
# Verify dataset loading
from ultralytics.data import YOLODataset
dataset = YOLODataset(img_path='train/images', data={'names': ['class1', 'class2']})
print(f"Dataset size: {len(dataset)}")

# Verify annotations
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('train/images/image1.jpg')
with open('train/labels/image1.txt', 'r') as f:
    labels = f.readlines()
    
# Draw bounding boxes
for label in labels:
    class_id, x_center, y_center, width, height = map(float, label.split())
    # Convert normalized coordinates to pixel coordinates
    h, w = img.shape[:2]
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

## Additional Resources

### Official Documentation
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [YOLO Paper (Original)](https://arxiv.org/abs/1506.02640)

### Datasets
- [COCO Dataset](https://cocodataset.org/)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [Roboflow Universe](https://universe.roboflow.com/) - Pre-labeled datasets

### Annotation Tools
- [Roboflow](https://roboflow.com/)
- [CVAT](https://github.com/opencv/cvat)
- [LabelImg](https://github.com/heartexlabs/labelImg)
- [Label Studio](https://labelstud.io/)
- [Makesense.ai](https://www.makesense.ai/)

### Tutorials and Guides
- [Ultralytics YOLOv8 Quickstart](https://docs.ultralytics.com/quickstart/)
- [Custom Dataset Training Guide](https://docs.ultralytics.com/modes/train/)
- [YOLO Model Export Guide](https://docs.ultralytics.com/modes/export/)

### Community
- [Ultralytics Discord](https://discord.com/invite/ultralytics)
- [Ultralytics GitHub Discussions](https://github.com/ultralytics/ultralytics/discussions)
- [Stack Overflow - YOLO Tag](https://stackoverflow.com/questions/tagged/yolo)

### Advanced Topics
- **Transfer Learning**: Fine-tuning pre-trained models
- **Hyperparameter Tuning**: Optimizing training parameters
- **Model Ensemble**: Combining multiple models
- **Real-time Inference**: Optimizing for speed
- **Edge Deployment**: Running on mobile/embedded devices

## Contributing

Contributions are welcome! If you find issues or have suggestions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [YOLO Community](https://github.com/ultralytics/ultralytics) for continuous improvements
- [Google Colab](https://colab.research.google.com/) for free GPU access

---

**Happy Training! 🚀**

For questions or support, please open an issue in this repository.
