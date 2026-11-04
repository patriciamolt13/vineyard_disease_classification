import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_KEYS = [
    "mobilevit_xxs",
    "tf_efficientnet_lite1",
    "mobilenetv2_100",
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
]

MODEL_NAMES = {
    "mobilevit_xxs": "MobileViT-XXS",
    "tf_efficientnet_lite1": "EfficientNet-Lite1",
    "mobilenetv2_100": "MobileNetV2",
    "mobilenetv3_small_100": "MobileNetV3-Small",
    "mobilenetv3_large_100": "MobileNetV3-Large",
}

CONFUSION_MATRIX_FILES = {
    "mobilevit_xxs": "confusion_matrix_mobilevit_xxs.csv",
    "tf_efficientnet_lite1": "confusion_matrix_efficientnetlite1.csv",
    "mobilenetv2_100": "confusion_matrix_mobilenetv2_100.csv",
    "mobilenetv3_small_100": "confusion_matrix_mobilenetv3_small_100.csv",
    "mobilenetv3_large_100": "confusion_matrix_mobilenetv3_large_100.csv",
}

OUTPUT_DIR = "graficos_confusion_matrices"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FONT SIZES
# ============================================================

AXIS_FONT_SIZE = 34
CLASS_FONT_SIZE = 28
NUMBER_FONT_SIZE = 34
COLORBAR_FONT_SIZE = 30

# ============================================================
# ABBREVIATED CLASS NAMES
# IMPORTANT: must follow the same order as the CSV columns
# ============================================================

CLASS_ABBREVIATIONS = {
    "black_rot": "BR",
    "downy_mildew": "DM",
    "esca": "E",
    "healthy": "H",
    "powdery_mildew": "PM",
}

# ============================================================
# CREATE CONFUSION MATRICES
# ============================================================

for model_key in MODEL_KEYS:

    model_name = MODEL_NAMES[model_key]
    cm_csv = CONFUSION_MATRIX_FILES[model_key]

    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.exists(cm_csv):
        print("Confusion matrix file not found:", cm_csv)
        continue

    # ========================================================
    # LOAD CONFUSION MATRIX
    # ========================================================

    cm_df = pd.read_csv(
        cm_csv,
        index_col=0
    )

    confusion_matrix = cm_df.to_numpy()

    # Create abbreviated labels from CSV column names
    class_names = [
        CLASS_ABBREVIATIONS.get(
            name.lower(),
            name
        )
        for name in cm_df.columns
    ]

    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    cm_display.plot(
        ax=ax,
        values_format="d",
        cmap="YlGn"
    )

    # ========================================================
    # AXIS LABELS
    # ========================================================

    ax.set_xlabel(
        "Predicted Class",
        fontsize=AXIS_FONT_SIZE
    )

    ax.set_ylabel(
        "True Class",
        fontsize=AXIS_FONT_SIZE
    )

    # ========================================================
    # CLASS ABBREVIATIONS
    # ========================================================

    plt.setp(
        ax.get_xticklabels(),
        rotation=0,
        fontsize=CLASS_FONT_SIZE
    )

    plt.setp(
        ax.get_yticklabels(),
        fontsize=CLASS_FONT_SIZE
    )

    # ========================================================
    # NUMBERS INSIDE CONFUSION MATRIX
    # ========================================================

    if cm_display.text_ is not None:

        for text in cm_display.text_.ravel():
            text.set_fontsize(NUMBER_FONT_SIZE)

    # ========================================================
    # COLOUR BAR
    # ========================================================

    if len(fig.axes) > 1:

        fig.axes[1].tick_params(
            labelsize=COLORBAR_FONT_SIZE
        )

    fig.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    file_base = f"cm_{model_key}"

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{file_base}.pdf"
        ),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{file_base}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Confusion matrix saved for: {model_name}")

# ============================================================
# FINISHED
# ============================================================

print("\nAll confusion matrices saved.")