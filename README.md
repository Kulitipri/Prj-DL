# Malware Classification using MobileNetV2

## 1. Introduction & Theoretical Foundation

### 1.1 Overview

This project utilizes Deep Learning to classify various types of malware based on their visual representations. Instead of direct source code analysis, this method converts malware binary files into grayscale images and employs a Convolutional Neural Network (CNN) for recognition.

### 1.2 Transfer Learning with MobileNetV2

**Transfer Learning** is a technique where a model pre-trained on a large dataset (ImageNet) is reused and fine-tuned for a specific task. This approach offers several benefits:

* **Reduced Training Time**: Leveraging pre-existing features.
* **Improved Accuracy**: Effective even with smaller datasets.
* **Feature Reusability**: Utilizes high-level features learned from millions of images.

**MobileNetV2** was chosen for its:

* **Lightweight Architecture**: Optimized for mobile and embedded systems.
* **Inverted Residuals**: Uses linear bottlenecks and inverted residual connections.
* **Depthwise Separable Convolutions**: Significantly reduces parameter count and computation.
* **Efficiency**: Provides an excellent balance between accuracy and computational cost.

### 1.3 Training Strategy

```
Strategy: Freeze Pre-trained Layers + Custom Classifier

┌─────────────────────────────┐
│    MobileNetV2 Base Model   │
│   (Frozen - weights locked) │ ← Pre-trained on ImageNet
│    150x150x3 → Features     │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  GlobalAveragePooling2D     │ ← Flatten spatial dimensions
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│   Dense(25, softmax)        │ ← Custom classifier for 25 malware classes
└─────────────────────────────┘

```

**Rationale for freezing the base model:**

* Preserves feature extractors learned from ImageNet.
* Trains only the classifier layer for malware-specific patterns.
* Prevents overfitting on smaller datasets.
* Drastically reduces training time.

---

## 2. Dataset

### 2.1 Data Source

* **Dataset**: Malimg Dataset via Kaggle.
* **Source**: `manmandes/malimg`.
* **Automation**: Downloaded using the `kagglehub` API.

### 2.2 Dataset Structure

The dataset is partitioned into three sets:

* **Training set**: For model optimization.
* **Validation set**: For monitoring performance during training.
* **Test set**: For final unbiased evaluation.

### 2.3 Characteristics

* **Classes**: 25 distinct malware families.
* **Format**: Grayscale images (PNG).
* **Input Size**: 150x150 pixels.
* **Class Imbalance**: Notable variation in the number of samples per class.

### 2.4 Preprocessing

```python
# Rescaling: Normalize pixel values from [0, 255] → [0, 1]
train_datagen = ImageDataGenerator(rescale=1/255)

# Load data with target_size=(150, 150)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

```

**Rationale for rescaling:**

* Neural networks converge faster with smaller input values.
* Ensures numerical stability during gradient descent.

---

## 3. System Requirements (Prerequisites)

### 3.1 Python Libraries

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub

```

### 3.2 Hardware Requirements

* **RAM**: Minimum 8GB (16GB recommended).
* **GPU**: Optional but strongly recommended (NVIDIA CUDA-compatible).
* **Storage**: ~2-3GB for the dataset and saved models.

### 3.3 Kaggle Setup

1. Create a Kaggle account.
2. Generate an API token from account settings.
3. Download `kaggle.json`.
4. Place it in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).

---

## 4. Model Architecture

### 4.1 Architecture Overview

```
Input Image (150x150x3)
        ↓
┌───────────────────────────────────────┐
│      MobileNetV2 Base (Frozen)        │
│  - Entry Flow                         │
│  - 17 Inverted Residual Blocks         │
│  - Output: 5x5x1280 feature maps      │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│       GlobalAveragePooling2D          │
│    5x5x1280 → 1280 features           │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│    Dense(25, activation='softmax')    │
│    Output: 25-class probabilities     │
└───────────────────────────────────────┘

```

### 4.2 Layer Details

* **Base Model**: MobileNetV2 with `weights='imagenet'`, `include_top=False`.
* **GlobalAveragePooling2D**: Reduces dimensions to 1280 features; more robust to spatial translations than flattening.
* **Dense Layer**: 1280 inputs to 25 outputs using Softmax.

### 4.3 Compilation & Loss

* **Optimizer**: Adam (Learning rate = 0.0001).
* **Loss**: Categorical Crossentropy.


* **Class Weights**: Applied to handle imbalance by penalizing errors in minority classes more heavily.

---

## 5. Workflow

### 5.1 Training Process

1. **Data Loading**: Automatically download and split data.
2. **Model Building**: Instantiate MobileNetV2 base and append custom layers.
3. **Training**: Run for 10 epochs using `class_weight` to handle imbalance.
4. **Evaluation**: Test performance on unseen data.
5. **Visualization**: Plot Accuracy/Loss curves and Confusion Matrix.
6. **Model Saving**: Export as `.keras` file.

### 5.2 Prediction Workflow

The system takes a raw malware image, rescales it, expands dimensions for batch processing, and outputs the predicted malware family name using `np.argmax` on the Softmax output.

---

## 6. Expected Results

### 6.1 Performance Metrics

* **Training Accuracy**: 85-95%.
* **Validation Accuracy**: 80-90%.
* **Test Accuracy**: 78-88%.

### 6.2 Comparison with Other Architectures

| Model | Parameters | Accuracy | Speed |
| --- | --- | --- | --- |
| **MobileNetV2 (Ours)** | **~2.3M** | **85-90%** | **Fast** |
| ResNet50 | ~25M | 88-92% | Medium |
| Custom CNN | <1M | 75-85% | Fast |
| VGG16 | ~138M | 87-91% | Slow |

### 6.3 Potential Issues & Solutions

* **Overfitting**: If Train Acc >> Val Acc, consider adding Dropout or Data Augmentation.
* **Underfitting**: If both are low, unfreeze the top layers of the base model for fine-tuning.

---

## 7. References

* [MobileNetV2 Paper: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
* [Malimg Dataset on Kaggle](https://www.kaggle.com/datasets/manmandes/malimg)

---

**Author**: Khai Nguyen Thien

**Date**: January 2026

**License**: MIT

Would you like me to help you draft the **Classification Report** analysis or write a **Python script** to automate the `kaggle.json` setup for your project?
