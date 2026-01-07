# Phân loại Malware sử dụng MobileNetV2

## 1. Giới thiệu & Cơ sở Lý thuyết

### 1.1 Tổng quan
Dự án này sử dụng Deep Learning để phân loại các loại malware (phần mềm độc hại) dựa trên hình ảnh visualization của chúng. Thay vì phân tích mã nguồn trực tiếp, phương pháp này chuyển đổi các file malware thành hình ảnh grayscale và sử dụng mô hình CNN để nhận diện.

### 1.2 Transfer Learning với MobileNetV2
**Transfer Learning** là kỹ thuật sử dụng một mô hình đã được pre-trained trên dataset lớn (ImageNet) và tinh chỉnh nó cho một tác vụ cụ thể. Điều này giúp:
- Giảm thời gian training
- Cải thiện độ chính xác với ít dữ liệu hơn
- Tận dụng các feature đã học từ millions of images

**MobileNetV2** được chọn vì:
- **Kiến trúc nhẹ**: Thiết kế tối ưu cho thiết bị di động và embedded systems
- **Inverted Residuals**: Sử dụng linear bottleneck và inverted residual connections
- **Depthwise Separable Convolutions**: Giảm số lượng parameters và tính toán
- **Hiệu quả**: Cân bằng tốt giữa accuracy và computational cost

### 1.3 Phương pháp Training
```
Strategy: Freeze Pre-trained Layers + Custom Classifier

┌─────────────────────────────┐
│   MobileNetV2 Base Model    │
│   (Frozen - weights locked) │ ← Pre-trained trên ImageNet
│   150x150x3 → Features      │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  GlobalAveragePooling2D     │ ← Flatten spatial dimensions
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  Dense(25, softmax)         │ ← Custom classifier cho 25 malware classes
└─────────────────────────────┘
```

**Lý do freeze base model:**
- Giữ lại các feature extractors đã học từ ImageNet
- Chỉ train classifier layer cho malware classification
- Tránh overfitting khi dataset nhỏ
- Giảm thời gian training đáng kể

## 2. Dataset (Bộ dữ liệu)

### 2.1 Nguồn dữ liệu
- **Dataset**: Malimg Dataset từ Kaggle
- **Source**: `manmandes/malimg`
- **Tải tự động**: Sử dụng `kagglehub` API

### 2.2 Cấu trúc Dataset
Dataset được chia thành 3 phần:
- **Training set**: Để huấn luyện mô hình
- **Validation set**: Để theo dõi performance trong quá trình training
- **Test set**: Để đánh giá cuối cùng

### 2.3 Đặc điểm
- **Số lượng classes**: 25 loại malware families
- **Định dạng**: Grayscale images (PNG)
- **Input size**: 150x150 pixels
- **Vấn đề class imbalance**: Dataset có sự chênh lệch số lượng samples giữa các classes

### 2.4 Preprocessing
```python
# Rescaling: Normalize pixel values từ [0, 255] → [0, 1]
train_datagen = ImageDataGenerator(rescale=1/255)

# Load data với target_size=(150, 150)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
```

**Lý do rescaling:**
- Neural networks hoạt động tốt hơn với input values nhỏ
- Giúp gradient descent converge nhanh hơn
- Tránh numerical instability

## 3. Yêu cầu Hệ thống (Prerequisites)

### 3.1 Python Libraries
```bash
pip install tensorflow
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install kagglehub
```

### 3.2 Hardware Requirements
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: Optional nhưng strongly recommended
  - CUDA-compatible GPU (NVIDIA)
  - Giảm training time từ hours → minutes
- **Storage**: ~2-3GB cho dataset

### 3.3 Kaggle Setup
1. Tạo Kaggle account
2. Generate API token từ account settings
3. Download `kaggle.json`
4. Place trong `~/.kaggle/` (Linux/Mac) hoặc `%USERPROFILE%\.kaggle\` (Windows)

### 3.4 Software
- **Python**: 3.8 hoặc cao hơn
- **TensorFlow**: 2.x
- **Jupyter Notebook** hoặc **JupyterLab**

## 4. Cấu trúc Mô hình CNN

### 4.1 Architecture Overview
```
Input Image (150x150x3)
        ↓
┌───────────────────────────────────────┐
│     MobileNetV2 Base (Frozen)         │
│  - Entry Flow                         │
│  - 17 Inverted Residual Blocks        │
│  - Expansion factors: 1, 6            │
│  - Output: 5x5x1280 feature maps      │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│   GlobalAveragePooling2D              │
│   5x5x1280 → 1280 features            │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│   Dense(25, activation='softmax')     │
│   Output: 25-class probabilities      │
└───────────────────────────────────────┘
```

### 4.2 Layer Details

#### Base Model: MobileNetV2
```python
base_model = MobileNetV2(
    weights='imagenet',      # Pre-trained weights
    include_top=False,       # Loại bỏ classification head
    input_shape=(150, 150, 3)
)

# Freeze tất cả layers
for layer in base_model.layers:
    layer.trainable = False
```

**Parameters:**
- Total params: ~2.3 million
- Trainable params trong base: 0 (frozen)

#### Custom Classifier
```python
# Pooling layer
x = GlobalAveragePooling2D()(base_model.output)

# Output layer
output = Dense(25, activation='softmax')(x)
```

**GlobalAveragePooling2D:**
- Reduce 5x5x1280 → 1280 vector
- Giảm overfitting so với Flatten + Dense
- Invariant to spatial translations

**Dense Layer:**
- 1280 inputs → 25 outputs
- Softmax activation cho multi-class classification
- Trainable params: 1280 × 25 + 25 = 32,025

### 4.3 Compilation
```python
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

**Optimizer: Adam**
- Adaptive learning rate cho mỗi parameter
- Combines momentum + RMSprop
- Learning rate = 0.0001 (conservative để tránh overfitting)

**Loss: Categorical Crossentropy**
- Standard cho multi-class classification
- Formula: -Σ(y_true * log(y_pred))

### 4.4 Class Weights
```python
class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(train_generator.classes), 
    y=train_generator.classes
)
```

**Tại sao cần class weights:**
- Dataset imbalanced → model bias về majority class
- Class weights penalize errors trên minority classes nhiều hơn
- Formula: weight = n_samples / (n_classes × n_samples_per_class)

## 5. Cách vận hành Code (Workflow)

### 5.1 Training Process Flow
```
Step 1: Data Loading & Preparation
   ↓
Step 2: Model Building
   ↓
Step 3: Training
   ↓
Step 4: Evaluation
   ↓
Step 5: Visualization & Analysis
   ↓
Step 6: Model Saving
```

### 5.2 Chi tiết từng bước

#### Step 1: Data Loading (Cells 0-6)
```python
# 1. Download dataset
path = kagglehub.dataset_download("manmandes/malimg")

# 2. Define directories
train_dir = os.path.join(path, "malimg_dataset/train")
val_dir = os.path.join(path, "malimg_dataset/val")
test_dir = os.path.join(path, "malimg_dataset/test")

# 3. Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), 
    batch_size=32, class_mode='categorical'
)
```

**Batch processing:**
- Batch size = 32: Load 32 images cùng lúc
- Tối ưu memory và computation
- Stable gradient updates

#### Step 2: Model Building (Cells 8-11)
```python
# 1. Load pre-trained base
base_model = MobileNetV2(weights='imagenet', 
                         include_top=False,
                         input_shape=(150, 150, 3))

# 2. Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# 3. Add custom top
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(25, activation='softmax')(x)

# 4. Create final model
model = Model(inputs=base_model.input, outputs=output)
```

#### Step 3: Training (Cell 14)
```python
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict
)
```

**Training dynamics:**
- **Epochs = 10**: Số lần model xem toàn bộ training data
- **Validation data**: Đánh giá sau mỗi epoch để detect overfitting
- **Class weights**: Adjust loss theo class distribution

**Mỗi epoch:**
```
For each batch in training data:
    1. Forward pass: Compute predictions
    2. Calculate loss (weighted by class_weights)
    3. Backward pass: Compute gradients
    4. Update weights (Adam optimizer)
    
After all batches:
    5. Evaluate on validation set
    6. Log metrics (accuracy, loss)
```

**Training history được lưu:**
```csv
epoch,train_accuracy,val_accuracy,train_loss,val_loss
1,0.xx,0.xx,x.xx,x.xx
...
```

#### Step 4: Evaluation (Cell 15)
```python
# Evaluate trên các sets
val_loss, val_accuracy = model.evaluate(val_generator)
train_loss, train_accuracy = model.evaluate(train_generator)
test_loss, test_accuracy = model.evaluate(test_generator)
```

**Metrics:**
- **Accuracy**: % predictions đúng
- **Loss**: Categorical crossentropy value

#### Step 5: Analysis (Cells 16-19)

**5a. Training curves:**
```python
plt.plot(history.history['accuracy'])      # Training accuracy
plt.plot(history.history['val_accuracy'])  # Validation accuracy
```
- Kiểm tra overfitting: Train accuracy >> Val accuracy
- Convergence: Metrics plateau

**5b. Classification Report:**
```python
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred_classes)
```
- Precision, Recall, F1-score cho mỗi class
- Identify classes mà model struggle

**5c. Confusion Matrix:**
```python
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True)
```
- Visual analysis của predictions
- Identify common misclassifications

#### Step 6: Model Saving (Cell 20)
```python
model.save('mobilenet_model.keras')
```
- Lưu full model (architecture + weights)
- Có thể load lại để inference

### 5.3 Prediction Workflow (Cells 22-27)

```python
# 1. Load model
model = tf.keras.models.load_model('mobilenet_model.keras')

# 2. Preprocess image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Rescale

# 3. Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
```

## 6. Kết quả (Dự kiến)

### 6.1 Expected Performance
Dựa trên MobileNetV2 architecture và dataset quality:

**Training Phase:**
- **Training Accuracy**: 85-95% sau 10 epochs
- **Validation Accuracy**: 80-90%
- **Training Loss**: Giảm dần từ ~3.2 → ~0.5
- **Validation Loss**: Giảm và stabilize ~0.8-1.2

**Test Set Performance:**
- **Test Accuracy**: 78-88%
- **Per-class F1-score**: Varies 0.65-0.95 tùy class

### 6.2 Convergence Pattern
```
Epoch 1:  Train Acc ~60%  | Val Acc ~55%  (Learning initial patterns)
Epoch 3:  Train Acc ~75%  | Val Acc ~70%  (Rapid improvement)
Epoch 5:  Train Acc ~85%  | Val Acc ~80%  (Slowing down)
Epoch 10: Train Acc ~90%  | Val Acc ~85%  (Near convergence)
```

### 6.3 Output Files
```
training_history_YYYYMMDD_HHMMSS.csv  - Training metrics log
mobilenet_model.keras                  - Saved model
```

### 6.4 Confusion Matrix Insights
- **Diagonal elements**: High values indicate good classification
- **Off-diagonal**: Common confusions giữa similar malware families
- **Balanced performance**: Class weights giúp improve minority classes

### 6.5 Potential Issues & Solutions

**Issue 1: Overfitting**
- **Symptom**: Train accuracy >> Val accuracy
- **Solution**: 
  - Add dropout layers
  - Increase data augmentation
  - Reduce model complexity

**Issue 2: Underfitting**
- **Symptom**: Both train & val accuracy low
- **Solution**:
  - Unfreeze một số top layers của base model
  - Train longer (more epochs)
  - Increase model capacity

**Issue 3: Class Imbalance Impact**
- **Symptom**: Poor performance on minority classes
- **Solution**:
  - Already applied: Class weights
  - Alternative: SMOTE, class-balanced sampling

### 6.6 Comparison với Other Approaches

| Model | Params | Accuracy | Speed |
|-------|--------|----------|-------|
| MobileNetV2 (Ours) | ~2.3M | 85-90% | Fast |
| ResNet50 | ~25M | 88-92% | Medium |
| Custom CNN | <1M | 75-85% | Fast |
| VGG16 | ~138M | 87-91% | Slow |

**MobileNetV2 advantages:**
- Excellent accuracy/efficiency tradeoff
- Suitable for deployment
- Fast inference time

## 7. Tài liệu tham khảo

### Papers
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Malware Detection using Deep Learning](https://ieeexplore.ieee.org)

### Documentation
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras Applications - MobileNetV2](https://keras.io/api/applications/mobilenet/)

### Dataset
- [Malimg Dataset on Kaggle](https://www.kaggle.com/datasets/manmandes/malimg)

---

**Project Info:**
- **Author**: [Your Name]
- **Date**: January 2026
- **Framework**: TensorFlow 2.x + Keras
- **License**: MIT

**Contact:**
- Email: [your-email]
- GitHub: [your-github]
