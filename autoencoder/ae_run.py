import os, numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

DATA_ROOT = r"D:\MalwareVision\dataset"
IMG_SIZE = (64, 64)
BATCH = 64
SEED = 42

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

# chọn NORMAL_CLASS: ưu tiên Allaple.A nếu có, không thì chọn class đầu
classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
NORMAL_CLASS = "Allaple.A" if "Allaple.A" in classes else classes[0]
print("NORMAL_CLASS =", NORMAL_CLASS)

def make_ds(dir_path, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        dir_path,
        labels=None,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
        seed=SEED
    )
    ds = ds.map(lambda x: tf.cast(x, tf.float32)/255.0, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

ds_train = make_ds(os.path.join(TRAIN_DIR, NORMAL_CLASS), shuffle=True)
ds_val   = make_ds(os.path.join(VAL_DIR, NORMAL_CLASS), shuffle=False)
ds_train_ae = ds_train.map(lambda x: (x, x)).prefetch(tf.data.AUTOTUNE)
ds_val_ae   = ds_val.map(lambda x: (x, x)).prefetch(tf.data.AUTOTUNE)
ds_test_normal = make_ds(os.path.join(TEST_DIR, NORMAL_CLASS), shuffle=False).take(30)

# gộp anomaly từ vài class khác để chạy nhanh
anom_classes = [c for c in os.listdir(TEST_DIR)
                if os.path.isdir(os.path.join(TEST_DIR, c)) and c != NORMAL_CLASS][:10]

ds_test_anom = None
for c in anom_classes:
    d = make_ds(os.path.join(TEST_DIR, c), shuffle=False).take(10)
    ds_test_anom = d if ds_test_anom is None else ds_test_anom.concatenate(d)

def build_ae(input_shape=(64,64,1)):
    inp = layers.Input(shape=input_shape)
    # Encoder
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    z = layers.MaxPooling2D(2, padding="same")(x)
    # Decoder
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(z)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    out = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    ae = Model(inp, out)
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return ae

def recon_scores(model, ds):
    scores = []
    for x in ds:
        xhat = model.predict(x, verbose=0)
        mse = np.mean((x.numpy() - xhat)**2, axis=(1,2,3))
        scores.append(mse)
    return np.concatenate(scores)

ae = build_ae((64,64,1))
hist = ae.fit(
    ds_train_ae,
    validation_data=ds_val_ae,
    epochs=20,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
)


# Hình 1: loss curve
plt.figure()
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("AE Loss Curve")
plt.legend()
plt.savefig("loss_curve.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved loss_curve.png")

# Threshold
val_scores = recon_scores(ae, ds_val)
threshold = np.percentile(val_scores, 95)
print("Threshold (p95 normal val) =", threshold)

# Hình 2: histogram
normal_scores = recon_scores(ae, ds_test_normal)
anom_scores   = recon_scores(ae, ds_test_anom)

plt.figure()
plt.hist(normal_scores, bins=50, alpha=0.7, label=f"Normal ({NORMAL_CLASS})")
plt.hist(anom_scores, bins=50, alpha=0.7, label="Anomaly (others)")
plt.axvline(threshold, linestyle="--", label="Threshold")
plt.xlabel("Reconstruction error (MSE)"); plt.ylabel("Count")
plt.title("Normal vs Anomaly Error")
plt.legend()
plt.savefig("score_hist.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved score_hist.png")

print("Normal flagged %:", (normal_scores > threshold).mean()*100)
print("Anomaly flagged %:", (anom_scores > threshold).mean()*100)

# Hình 3: recon examples
def one_batch(ds):
    for x in ds.take(1): return x

xN = one_batch(ds_test_normal)
xNhat = ae.predict(xN, verbose=0)

xA = one_batch(ds_test_anom)
xAhat = ae.predict(xA, verbose=0)

i = 0
fig = plt.figure(figsize=(10,6))
diffN = np.abs(xN[i].numpy() - xNhat[i])
plt.subplot(2,3,1); plt.imshow(xN[i].numpy().squeeze(), cmap="gray"); plt.title("Normal: Original"); plt.axis("off")
plt.subplot(2,3,2); plt.imshow(xNhat[i].squeeze(), cmap="gray"); plt.title("Normal: Recon"); plt.axis("off")
plt.subplot(2,3,3); plt.imshow(diffN.squeeze(), cmap="gray"); plt.title("Normal: |Diff|"); plt.axis("off")

diffA = np.abs(xA[i].numpy() - xAhat[i])
plt.subplot(2,3,4); plt.imshow(xA[i].numpy().squeeze(), cmap="gray"); plt.title("Anomaly: Original"); plt.axis("off")
plt.subplot(2,3,5); plt.imshow(xAhat[i].squeeze(), cmap="gray"); plt.title("Anomaly: Recon"); plt.axis("off")
plt.subplot(2,3,6); plt.imshow(diffA.squeeze(), cmap="gray"); plt.title("Anomaly: |Diff|"); plt.axis("off")

plt.tight_layout()
plt.savefig("recon_examples.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved recon_examples.png")

print("\nDONE -> loss_curve.png, score_hist.png, recon_examples.png")
