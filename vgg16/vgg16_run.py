import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# ====== CONFIG ======
DATA_ROOT = r"D:\MalwareVision\dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

IMG_SIZE = (224, 224)   # VGG16 expects 224x224
BATCH = 32
SEED = 42
EPOCHS = 10

OUT_DIR = r"D:\MalwareVision\vgg16_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== DATASET ======
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    seed=SEED,
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    seed=SEED,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    seed=SEED,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Num classes =", num_classes)

# VGG16 preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16

AUTOTUNE = tf.data.AUTOTUNE
train_ds_pp = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds_pp   = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds_pp  = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ====== CLASS WEIGHTS (đỡ lệch dữ liệu) ======
# tính số ảnh mỗi class trong TRAIN_DIR
counts = []
for c in class_names:
    cdir = os.path.join(TRAIN_DIR, c)
    n = 0
    for fn in os.listdir(cdir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            n += 1
    counts.append(n)

counts = np.array(counts, dtype=np.float32)
total = counts.sum()
class_weight = {i: float(total / (num_classes * counts[i])) for i in range(num_classes)}
print("Class weights sample:", dict(list(class_weight.items())[:3]))

# ====== MODEL (Transfer Learning) ======
base = VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False  # freeze backbone

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
]

hist = model.fit(
    train_ds_pp,
    validation_data=val_ds_pp,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

# ====== PLOTS: train/val curves ======
plt.figure()
plt.plot(hist.history["accuracy"], label="train_acc")
plt.plot(hist.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("VGG16 Accuracy Curve")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "vgg16_acc_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("VGG16 Loss Curve")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "vgg16_loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

# ====== TEST EVAL ======
test_loss, test_acc = model.evaluate(test_ds_pp, verbose=0)
print("TEST accuracy =", test_acc, " | loss =", test_loss)

# ====== CONFUSION MATRIX ======
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds_pp, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

# plot confusion matrix (normalized)
cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
plt.figure(figsize=(10, 8))
plt.imshow(cm_norm, interpolation="nearest")
plt.title("VGG16 Confusion Matrix (Normalized)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUT_DIR, "vgg16_confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

# ====== SAVE MODEL + REPORT ======
model_path = os.path.join(OUT_DIR, "vgg16_malimg.keras")
model.save(model_path)

with open(os.path.join(OUT_DIR, "vgg16_results.txt"), "w", encoding="utf-8") as f:
    f.write(f"TEST accuracy: {test_acc}\n")
    f.write(f"TEST loss: {test_loss}\n")
    f.write("Class names:\n")
    f.write("\n".join(class_names) + "\n")

print("\nDONE. Outputs in:", OUT_DIR)
print("Saved:", model_path)
