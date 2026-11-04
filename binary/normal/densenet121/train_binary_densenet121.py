import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# 0) Reproducibility
# =========================
SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# =========================
# 1) Settings
# =========================
TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"

OUTPUT_DIR = "outputs_blackrot_densenet121"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_HEAD = 10
LR_HEAD = 1e-4

EPOCHS_FINETUNE = 5
LR_FINETUNE = 1e-5
UNFREEZE_LAST_N = 30

USE_CLASS_WEIGHTS = True

# =========================
# 2) Load datasets
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=SEED,
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=SEED,
    shuffle=False
)

class_names = train_ds.class_names
print("Class names:", class_names)

if "black_rot" not in class_names or "non_black_rot" not in class_names:
    raise ValueError(f"Expected 'black_rot' and 'non_black_rot'. Found: {class_names}")

black_idx = class_names.index("black_rot")

# =========================
# 3) Force positive class = Black Rot (y=1)
# =========================
def remap_to_blackrot_positive(images, labels):
    labels = tf.cast(labels, tf.int32)
    y = tf.where(tf.equal(labels, black_idx), 1, 0)
    y = tf.cast(y, tf.float32)
    return images, tf.expand_dims(y, axis=-1)

train_ds = train_ds.map(remap_to_blackrot_positive, num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = test_ds.map(remap_to_blackrot_positive,  num_parallel_calls=tf.data.AUTOTUNE)

# =========================
# 4) Augmentation + Preprocess
# =========================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.05, seed=SEED),
    layers.RandomZoom(0.10, seed=SEED),
], name="augmentation")

# preprocess DenseNet
preprocess_input = tf.keras.applications.densenet.preprocess_input

def prep(images, labels):
    images = preprocess_input(images)  # só aqui
    return images, labels

train_ds = train_ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = test_ds.map(prep,  num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# =========================
# 5) Class weights (optional)
# =========================
class_weights = None
if USE_CLASS_WEIGHTS:
    y_train = []
    for _, y in train_ds.unbatch():
        y_train.append(int(y.numpy()[0]))
    y_train = np.array(y_train)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    total = n_neg + n_pos

    w0 = total / (2.0 * n_neg + 1e-12)
    w1 = total / (2.0 * n_pos + 1e-12)
    class_weights = {0: float(w0), 1: float(w1)}

    print(f"Train distribution -> neg(non_black_rot)={n_neg}, pos(black_rot)={n_pos}")
    print("Class weights:", class_weights)

# =========================
# 6) Build model (DenseNet121 backbone + binary head)
# =========================
base_model = tf.keras.applications.DenseNet121(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # head phase

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs, name="BlackRot_DenseNet121")
model.summary()

# =========================
# 7) Compile + callbacks (2 CSVs)
# =========================
metrics = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

best_path = os.path.join(OUTPUT_DIR, "densenet121_best.keras")

csv_log_head = os.path.join(OUTPUT_DIR, "training_log_head.csv")
csv_log_ft   = os.path.join(OUTPUT_DIR, "training_log_finetune.csv")

callbacks_head = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
    keras.callbacks.CSVLogger(csv_log_head),
]

callbacks_ft = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True),
    keras.callbacks.CSVLogger(csv_log_ft),
]

# =========================
# 8) Train Phase 1 (head)
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss="binary_crossentropy",
    metrics=metrics
)

print("\n=== Phase 1: Train head (backbone frozen) ===")
history1 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks_head
)

# =========================
# 9) Train Phase 2 (fine-tuning)
# =========================
print("\n=== Phase 2: Fine-tuning ===")
base_model.trainable = True

if UNFREEZE_LAST_N is not None and UNFREEZE_LAST_N > 0:
    for layer in base_model.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_FINETUNE),
    loss="binary_crossentropy",
    metrics=metrics
)

history2 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weights,
    callbacks=callbacks_ft
)

# =========================
# 10) Evaluate + Report
# =========================
print("\n=== Final evaluation on test set ===")
test_metrics = model.evaluate(test_ds, verbose=0)
print(dict(zip(model.metrics_names, test_metrics)))

y_true, y_prob = [], []
for xb, yb in test_ds:
    probs = model.predict(xb, verbose=0).reshape(-1)
    y_prob.extend(probs.tolist())
    y_true.extend(yb.numpy().reshape(-1).tolist())

y_true = np.array(y_true, dtype=int)
y_prob = np.array(y_prob, dtype=float)
y_pred = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
print("\nClassification Report (pos=black_rot):\n")
print(classification_report(
    y_true, y_pred,
    target_names=["non_black_rot", "black_rot"],
    digits=4
))

precision = tp / (tp + fp + 1e-12)
recall    = tp / (tp + fn + 1e-12)
f1        = 2 * precision * recall / (precision + recall + 1e-12)
acc       = (tp + tn) / (tp + tn + fp + fn + 1e-12)

results = {
    "model": "DenseNet121",
    "img_size": IMG_SIZE,
    "threshold": 0.5,
    "accuracy": float(acc),
    "precision_black_rot": float(precision),
    "recall_black_rot": float(recall),
    "f1_black_rot": float(f1),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "keras_eval": dict(zip(model.metrics_names, [float(x) for x in test_metrics])),
    "class_weights": class_weights,
    "training_logs": {"head": "training_log_head.csv", "finetune": "training_log_finetune.csv"},
}

print("\nFinal metrics JSON:\n", json.dumps(results, indent=2))

with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

np.save(os.path.join(OUTPUT_DIR, "confusion_matrix.npy"), cm)
np.save(os.path.join(OUTPUT_DIR, "y_true.npy"), y_true)
np.save(os.path.join(OUTPUT_DIR, "y_prob.npy"), y_prob)

final_model_path = os.path.join(OUTPUT_DIR, "densenet121_final.keras")
model.save(final_model_path)

print("\nSaved:")
print(" -", best_path)
print(" -", final_model_path)
print(" -", csv_log_head)
print(" -", csv_log_ft)
print(" - final_metrics.json + NPY files in", OUTPUT_DIR)
