import os
import copy
import time
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

import timm

# =========================
# CONFIG
# =========================
DATA_DIR = "dataset2"

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = 5

EPOCHS_HEAD = 10
EPOCHS_FINE = 50

EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 2e-3

LR_HEAD = 1e-3
LR_FINE = 1e-5

MODEL_NAME = "mobilenetv3_large_100"  # MobileNetV3-Large
MODEL_SAVE_PATH = "mobilenetv3_large_100.pth"
HISTORY_CSV = "training_history_mobilenetv3_large_100.csv"
CONFUSION_MATRIX_CSV = "confusion_matrix_mobilenetv3_large_100.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} device")

# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASETS
# =========================
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=eval_transform
)

if len(train_dataset.classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, found {len(train_dataset.classes)}: "
        f"{train_dataset.classes}"
    )

# Sanity check: class ordering must match across splits, since
# ImageFolder assigns indices alphabetically per-directory.
assert train_dataset.classes == val_dataset.classes == test_dataset.classes, (
    "Class order mismatch across splits! "
    f"train={train_dataset.classes}, val={val_dataset.classes}, "
    f"test={test_dataset.classes}"
)
class_names = train_dataset.classes

print("Classes:", class_names)
print("class_to_idx:", train_dataset.class_to_idx)
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))


# =========================
# DATALOADERS
# =========================
generator = torch.Generator()
generator.manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# CLASS WEIGHTS
# =========================
targets = np.array(train_dataset.targets)
counts = np.bincount(targets, minlength=NUM_CLASSES).astype(np.float32)

weights = counts.sum() / (NUM_CLASSES * (counts + 1e-12))
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

print("Class counts:", counts)
print("Class weights:", class_weights.cpu().numpy())


# =========================
# MODEL
# =========================
model = timm.create_model(MODEL_NAME, pretrained=True,num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# =========================
# FREEZE / UNFREEZE FUNCTIONS
# =========================
def freeze_backbone_train_classifier(model):
    for param in model.parameters():
        param.requires_grad = False

    classifier_modules = []

    # Try timm classifier accessor
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()

        if isinstance(classifier, nn.Module):
            classifier_modules.append(classifier)

        elif isinstance(classifier, (list, tuple)):
            classifier_modules.extend(
                [m for m in classifier if isinstance(m, nn.Module)]
            )

     # Try common classifier names
    for attr_name in ["classifier", "head", "fc"]:
        if hasattr(model, attr_name):
            module = getattr(model, attr_name)

            if isinstance(module, nn.Module):
                classifier_modules.append(module)

    # Remove duplicates
    unique_modules = []
    seen_ids = set()

    for module in classifier_modules:
        if id(module) not in seen_ids:
            unique_modules.append(module)
            seen_ids.add(id(module))

    if len(unique_modules) == 0:
        raise AttributeError(
            "Could not find classifier/head. Run print(model) to inspect the architecture."
        )

    # Unfreeze classifier
    for module in unique_modules:
        for param in module.parameters():
            param.requires_grad = True

def unfreeze_full_model(model):
    for param in model.parameters():
        param.requires_grad = True

def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def print_trainable_parameters(model, stage_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== {stage_name} ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = running_loss / max(total, 1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    return avg_loss, acc, prec, rec, f1, cm, y_true, y_pred


# =========================
# TRAIN ONE EPOCH
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, freeze_bn=False):

    model.train()
    # Keep BatchNorm statistics frozen
    # during Stage 1
    if freeze_bn:
        set_batchnorm_eval(model)

    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    return epoch_loss, epoch_acc

# =========================
# TWO-STAGE TRAINING FUNCTION
# =========================
def train_two_stage(model):
    since = time.time()

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())

    history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_prec_macro": [],
        "val_rec_macro": [],
        "val_f1_macro": [],
        "lr": []
    }

    epoch_counter = 0

    # =====================================================
    # STAGE 1: TRAIN CLASSIFIER ONLY
    # =====================================================
    print("\n==============================")
    print("STAGE 1: Training classifier head")
    print("==============================")

    freeze_backbone_train_classifier(model)
    print_trainable_parameters(model, "Stage 1 - classifier only")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD
    )

    for epoch in range(EPOCHS_HEAD):
        epoch_counter += 1

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            freeze_bn=True
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch_counter)
        history["stage"].append("classifier")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_prec_macro"].append(val_prec)
        history["val_rec_macro"].append(val_rec)
        history["val_f1_macro"].append(val_f1)
        history["lr"].append(current_lr)

        print(
            f"[Classifier] Epoch [{epoch + 1}/{EPOCHS_HEAD}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

    # =====================================================
    # STAGE 2: FINE-TUNE FULL MODEL
    # =====================================================
    print("\n==============================")
    print("STAGE 2: Fine-tuning full model")
    print("==============================")

    model.load_state_dict(best_model_weights)
    unfreeze_full_model(model)
    print_trainable_parameters(model, "Stage 2 - full fine-tuning")

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR_FINE
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    epochs_no_improve = 0
    stage_best_loss = float("inf")

    for epoch in range(EPOCHS_FINE):
        epoch_counter += 1

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            freeze_bn=False
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch_counter)
        history["stage"].append("fine_tuning")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_prec_macro"].append(val_prec)
        history["val_rec_macro"].append(val_rec)
        history["val_f1_macro"].append(val_f1)
        history["lr"].append(current_lr)

        # Update LR for the NEXT epoch
        scheduler.step(val_loss)

        print(
            f"[Fine-tuning] Epoch [{epoch + 1}/{EPOCHS_FINE}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
        )

        if val_loss < best_val_loss - MIN_DELTA:
             best_val_loss = val_loss
             best_model_weights = copy.deepcopy(model.state_dict())
 
        # Early stopping for Stage 2 (based on val_loss trend only)
        if val_loss < stage_best_loss - MIN_DELTA:
            stage_best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
 
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping in Stage 2.")
            break

    # =========================
    # TRAINING COMPLETE
    # =========================

    time_elapsed = time.time() - since

    print("\n=== Training complete ===")
    print(f"Training time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

    model.load_state_dict(best_model_weights)

    return model, history


# =========================
# TRAIN MODEL
# =========================
model, history = train_two_stage(model)


# =========================
# SAVE TRAINING HISTORY
# =========================
with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(list(history.keys()))

    for k in range(len(history["epoch"])):
        writer.writerow([history[col][k] for col in history.keys()])

print(f"\nSaved training history to: {HISTORY_CSV}")


# =========================
# FINAL TEST
# =========================
test_loss, test_acc, test_prec, test_rec, test_f1, test_cm, y_true, y_pred = evaluate(
    model,
    test_loader,
    criterion,
    DEVICE
)

print("\n=== Test Results ===")
print(f"Test Loss : {test_loss:.4f}")
print(f"Accuracy  : {test_acc:.4f}")
print(f"Precision : {test_prec:.4f}")
print(f"Recall    : {test_rec:.4f}")
print(f"F1-score  : {test_f1:.4f}")

print("\nConfusion Matrix:")
print(test_cm)


# =========================
# CLASSIFICATION REPORT
# =========================
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0
    )
)


# =========================
# SAVE CONFUSION MATRIX
# =========================
with open(CONFUSION_MATRIX_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true/pred"] + class_names)

    for i, row in enumerate(test_cm):
        writer.writerow([class_names[i]] + list(row))

print(f"Saved confusion matrix to: {CONFUSION_MATRIX_CSV}")


# =========================
# NUMBER OF PARAMETERS
# =========================
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n=== Model Complexity ===")
print(f"Total parameters: {num_params:,}")
print(f"Trainable parameters: {num_trainable:,}")


# =========================
# INFERENCE TIME FOR ONE IMAGE
# =========================
model.eval()

images, _ = next(iter(test_loader))
images = images.to(DEVICE)
single_image = images[0].unsqueeze(0)

with torch.no_grad():
    for _ in range(10):
        _ = model(single_image)

N = 100

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

start = time.time()

with torch.no_grad():
    for _ in range(N):
        _ = model(single_image)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

end = time.time()

avg_time = (end - start) / N

print("\n=== Inference Speed ===")
print(f"Average inference time per image: {avg_time * 1000:.2f} ms")
print(f"FPS: {1 / avg_time:.2f}")


# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")

model_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)

print(f"Model size: {model_size:.2f} MB")