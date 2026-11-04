import math
import os
import copy
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import timm


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset2"

MODEL_CONFIGS = {
    "efficientnetlite1": {
        "model_name": "tf_efficientnet_lite1",
        "model_path": "efficientnetlite1.pth",
        "img_size": 240,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },

    "mobilenetv2": {
        "model_name": "mobilenetv2_100",
        "model_path": "mobilenetv2_100.pth",
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },

    "mobilenetv3_small": {
        "model_name": "mobilenetv3_small_100",
        "model_path": "mobilenetv3_small_100.pth",
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },

    "mobilenetv3_large": {
        "model_name": "mobilenetv3_large_100",
        "model_path": "mobilenetv3_large_100.pth",
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

# Models
#MODEL_KEY = "efficientnetlite1"
#MODEL_KEY = "mobilenetv2"
#MODEL_KEY = "mobilenetv3_small"
MODEL_KEY = "mobilenetv3_large"

cfg = MODEL_CONFIGS[MODEL_KEY]

MODEL_NAME = cfg["model_name"]
MODEL_PATH = cfg["model_path"]
IMG_SIZE = cfg["img_size"]

NUM_CLASSES = 5
BATCH_SIZE = 32

# Six perturbation groups
MODES = ["early", "mid", "late", "classifier", "depthwise", "full"]

# Noise levels
EPSILONS = [0.0, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

N_RUNS = 100

OUTPUT_DIR = os.path.join("robustness_noise_tese", MODEL_KEY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_KEY}_uniform_noise_results.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_KEY}_uniform_noise_summary.csv"
)

LAYERS_CSV = os.path.join(
    OUTPUT_DIR,
    f"{MODEL_KEY}_selected_layers.csv"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("MODEL:", MODEL_KEY)
print("TIMM NAME:", MODEL_NAME)
print("MODEL PATH:", MODEL_PATH)
print("IMAGE SIZE:", IMG_SIZE)
print("DEVICE:", DEVICE)

# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================
# DATA
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=cfg["mean"],
        std=cfg["std"]
    )
])

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_transform
)

if len(test_dataset.classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, found {len(test_dataset.classes)}: "
        f"{test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print("Classes:", test_dataset.classes)
print("Test size:", len(test_dataset))


# ============================================================
# PRE-LOAD TEST SET INTO MEMORY
# ============================================================

print("\nLoading test set into memory...")

_all_images = []
_all_labels = []

for images, labels in test_loader:
    _all_images.append(images)
    _all_labels.append(labels)

CACHED_IMAGES = torch.cat(_all_images, dim=0).to(DEVICE)
CACHED_LABELS = torch.cat(_all_labels, dim=0).to(DEVICE)

del _all_images, _all_labels

print(f"Cached {CACHED_IMAGES.shape[0]} images on {DEVICE}.")

# ============================================================
# LOAD CLEAN TRAINED MODEL
# ============================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

clean_state_dict = copy.deepcopy(model.state_dict())

print("\nClean model loaded successfully.")

# ============================================================
# BUILD COMPARABLE LAYER GROUPS
# ============================================================

def perturbable_layer(module):
    # Layers with main learnable weights 

    return isinstance(module, (nn.Conv2d, nn.Linear))

def depthwise_conv(module):
    # Depthwise convolution

    return (
        isinstance(module, nn.Conv2d)
        and module.groups == module.in_channels
        and module.groups > 1
        and module.out_channels % module.in_channels == 0 # out_channels = K × in_channels
    )

def build_layer_groups(model):
    """
    Build six perturbation groups:
    early      -> first third of backbone Conv/Linear layers
    mid        -> middle third of backbone Conv/Linear layers
    late       -> final third of backbone Conv/Linear layers
    classifier -> final Linear layer
    depthwise  -> all depthwise convolution layers
    full       -> all Conv/Linear layers
    """

    all_layers = []
    linear_layers = []
    depthwise_layers = []

    for name, module in model.named_modules():

        if not perturbable_layer(module):
            continue

        # verify if the layers have weights
        if not hasattr(module, "weight") or module.weight is None:
            continue

        all_layers.append(name)

        if isinstance(module, nn.Linear):
            linear_layers.append(name)

        if depthwise_conv(module):
            depthwise_layers.append(name)

    if len(all_layers) == 0:
        raise RuntimeError("No Conv2d or Linear layers found.")

    if len(linear_layers) == 0:
        raise RuntimeError("No Linear layer found for classifier/head.")

    # final Linear layer as the classifier/head
    classifier_layers = [linear_layers[-1]]

    # Backbone excludes the final classifier
    backbone_layers = [
        name for name in all_layers
        if name not in classifier_layers
    ]

    n = len(backbone_layers)

    if n < 3:
        raise RuntimeError("Can't split into early/mid/late.")

    early_layers = backbone_layers[: n // 3]
    mid_layers = backbone_layers[n // 3 : 2 * n // 3]
    late_layers = backbone_layers[2 * n // 3 :]

    layer_groups = {
        "early": set(early_layers),
        "mid": set(mid_layers),
        "late": set(late_layers),
        "classifier": set(classifier_layers),
        "depthwise": set(depthwise_layers),
        "full": set(all_layers),
    }

    return layer_groups

LAYER_GROUPS = build_layer_groups(model)

def should_add_noise(name, mode):

    if mode not in LAYER_GROUPS:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    return name in LAYER_GROUPS[mode]

def count_perturbed_parameters(model, mode):

    total = 0
    layers = []

    for name, module in model.named_modules():

        if not should_add_noise(name, mode):
            continue

        if not hasattr(module, "weight") or module.weight is None:
            continue

        n_params = module.weight.numel()
        total += n_params

        layers.append({
            "model_key": MODEL_KEY,
            "model_name": MODEL_NAME,
            "mode": mode,
            "layer_name": name,
            "layer_type": module.__class__.__name__,
            "weight_shape": tuple(module.weight.shape),
            "n_params": n_params,
        })

    return total, layers

print("\nSelected layers for perturbation:")

all_layer_rows = []

for mode in MODES:
    total, layers = count_perturbed_parameters(model, mode)

    print("\nMode:", mode)
    print("Number of layers:", len(layers))
    print("Number of perturbed weight parameters:", f"{total:,}")

    if total == 0:
        raise RuntimeError(f"No layers selected for mode: {mode}")

    all_layer_rows.extend(layers)

pd.DataFrame(all_layer_rows).to_csv(LAYERS_CSV, index=False)

print("\nSaved selected layer list to:", LAYERS_CSV)

# ============================================================
# ADD UNIFORM NOISE AND MEASURE REAL PERTURBATION
# ============================================================

def add_noise(model, epsilon, mode):
    """
    Adds zero-mean uniform noise to selected model weights.

    ΔW = U(-1,1) * (sigma_W * ε *  sqrt{3})
    W' = W + ΔW 

    where:
        W       = original weight tensor
        W'      = perturbed weight tensor
        ΔW      = injected perturbation
        sigma_W = standard deviation of W
        ε       = relative perturbation level
    """

    sum_abs_delta = 0.0
    sum_sq_delta = 0.0
    sum_sq_weight = 0.0
    n_values = 0

    with torch.no_grad():

        for name, module in model.named_modules():

            if not should_add_noise(name, mode):
                continue

            if not hasattr(module, "weight") or module.weight is None:
                continue

            weight = module.weight

            if weight.numel() < 2:
                continue

            original_weight = weight.clone()
            weight_std = weight.float().std(unbiased=False)

            if weight_std.item() == 0 or torch.isnan(weight_std).item():
                continue

            if epsilon > 0:

                scale = weight_std * epsilon * math.sqrt(3)
                noise = (torch.rand_like(weight) * 2.0 - 1.0) * scale
            
                weight.add_(noise)

            delta = weight - original_weight

            sum_abs_delta += delta.abs().sum().item()
            sum_sq_delta += (delta.float() ** 2).sum().item()
            sum_sq_weight += (original_weight.float() ** 2).sum().item()
            n_values += delta.numel()

    if n_values > 0:
        mean_abs_delta = sum_abs_delta / n_values
        rms_delta = (sum_sq_delta / n_values) ** 0.5
        rms_weight = (sum_sq_weight / n_values) ** 0.5
    else:
        mean_abs_delta = 0.0
        rms_delta = 0.0
        rms_weight = 0.0

    if rms_weight > 0:
        relative_rms_noise = rms_delta / rms_weight
    else:
        relative_rms_noise = 0.0

    # SNR
    if sum_sq_delta > 0 and sum_sq_weight > 0:
        snr = rms_weight / rms_delta
        snr_db = 20.0 * np.log10(snr)
    else:
        snr = np.nan
        snr_db = np.nan

    return {
        "mean_abs_delta": mean_abs_delta,
        "rms_delta": rms_delta,
        "rms_weight": rms_weight,
        "relative_rms_noise": relative_rms_noise,
        "snr": snr,
        "snr_db": snr_db
    }

# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate(model, images=CACHED_IMAGES, labels=CACHED_LABELS, batch_size=BATCH_SIZE):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for start in range(0, images.shape[0], batch_size):

            end = start + batch_size

            batch_images = images[start:end]
            batch_labels = labels[start:end]

            outputs = model(batch_images)
            preds = outputs.argmax(dim=1)

            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return acc, prec, rec, f1

# ============================================================
# ROBUSTNESS TEST
# ============================================================

print("\nEvaluating clean model once...")

model.load_state_dict(clean_state_dict)
clean_acc, clean_prec, clean_rec, clean_f1 = evaluate(model)

print(
    f"Clean | Acc: {clean_acc:.4f} | "
    f"Prec: {clean_prec:.4f} | "
    f"Rec: {clean_rec:.4f} | "
    f"F1: {clean_f1:.4f}"
)

results = []
header_written = False

def append_result(row):
    global header_written

    df_row = pd.DataFrame([row])

    df_row.to_csv(
        RESULTS_CSV,
        mode="a" if header_written else "w",
        header=not header_written,
        index=False
    )

    header_written = True

for mode in MODES:

    for epsilon in EPSILONS:

        print("\n" + "-" * 80)
        print(f"Uniform noise | Mode: {mode} | Epsilon: {epsilon}")
        print("-" * 80)

        if epsilon == 0.0:

            model.load_state_dict(clean_state_dict)

            noise_info = add_noise(
                model=model,
                epsilon=0.0,
                mode=mode
            )

            row = {
                "model_key": MODEL_KEY,
                "model_name": MODEL_NAME,
                "mode": mode,
                "epsilon": epsilon,
                "run": 0,

                "mean_abs_delta": noise_info["mean_abs_delta"],
                "rms_delta": noise_info["rms_delta"],
                "rms_weight": noise_info["rms_weight"],
                "relative_rms_noise": noise_info["relative_rms_noise"],
                "snr": noise_info["snr"],
                "snr_db": noise_info["snr_db"],

                "accuracy": clean_acc,
                "precision_macro": clean_prec,
                "recall_macro": clean_rec,
                "f1_macro": clean_f1
            }

            results.append(row)
            append_result(row)

            print(
                f"Clean reused | "
                f"RMS weight: {noise_info['rms_weight']:.6f} | "
                f"SNR: N/A | "
                f"SNR (dB): N/A | "
                f"Acc: {clean_acc:.4f} | "
                f"Prec: {clean_prec:.4f} | "
                f"Rec: {clean_rec:.4f} | "
                f"F1: {clean_f1:.4f}"
            )

            continue

        run_start_time = time.time()

        for run in range(N_RUNS):

            # Reset model to clean weights before each run
            model.load_state_dict(clean_state_dict)

            run_seed = (
                42
                + run
                + 1000 * EPSILONS.index(epsilon)
                + 10000 * MODES.index(mode)
            )

            set_seed(run_seed)

            noise_info = add_noise(
                model=model,
                epsilon=epsilon,
                mode=mode
            )

            acc, prec, rec, f1 = evaluate(model)

            row = {
                "model_key": MODEL_KEY,
                "model_name": MODEL_NAME,
                "mode": mode,
                "epsilon": epsilon,
                "run": run,

                "mean_abs_delta": noise_info["mean_abs_delta"],
                "rms_delta": noise_info["rms_delta"],
                "rms_weight": noise_info["rms_weight"],
                "relative_rms_noise": noise_info["relative_rms_noise"],
                "snr": noise_info["snr"],
                "snr_db": noise_info["snr_db"],

                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1
            }

            results.append(row)
            append_result(row)

            if (run + 1) % 10 == 0 or (run + 1) == N_RUNS:

                elapsed = time.time() - run_start_time
                time_per_run = elapsed / (run + 1)

                print(
                    f"{run + 1}/{N_RUNS} runs | "
                    f"{time_per_run:.2f}s/run | "
                    f"SNR: {noise_info['snr']:.2f} | "
                    f"SNR (dB): {noise_info['snr_db']:.2f} dB | "
                    f"Acc: {acc:.4f} | "
                    f"F1: {f1:.4f}"
                )


# ============================================================
# RESULTS SUMMARY
# ============================================================

df = pd.DataFrame(results)

summary = (
    df.groupby(["model_key", "model_name", "mode", "epsilon"])
    .agg(
        mean_abs_delta_mean=("mean_abs_delta", "mean"),
        mean_abs_delta_std=("mean_abs_delta", "std"),

        rms_delta_mean=("rms_delta", "mean"),
        rms_delta_std=("rms_delta", "std"),

        rms_weight_mean=("rms_weight", "mean"),
        rms_weight_std=("rms_weight", "std"),

        relative_rms_noise_mean=("relative_rms_noise", "mean"),
        relative_rms_noise_std=("relative_rms_noise", "std"),

        snr_mean=("snr", "mean"),
        snr_std=("snr", "std"),

        snr_db_mean=("snr_db", "mean"),
        snr_db_std=("snr_db", "std"),

        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),

        precision_mean=("precision_macro", "mean"),
        precision_std=("precision_macro", "std"),

        recall_mean=("recall_macro", "mean"),
        recall_std=("recall_macro", "std"),

        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),

        n_runs=("run", "count")
    )
    .reset_index()
)

std_columns = [
    "mean_abs_delta_std",
    "rms_delta_std",
    "rms_weight_std",
    "relative_rms_noise_std",
    "snr_std",
    "snr_db_std",
    "accuracy_std",
    "precision_std",
    "recall_std",
    "f1_std",
]

summary[std_columns] = summary[std_columns].fillna(0)

summary.to_csv(SUMMARY_CSV, index=False)

print("Saved detailed results:", RESULTS_CSV)
print("Saved summary results:", SUMMARY_CSV)
print("Saved selected layers:", LAYERS_CSV)