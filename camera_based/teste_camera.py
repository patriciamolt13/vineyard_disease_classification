import numpy as np
import os
import time
import cv2
from picamera2 import Picamera2, Preview
from hailo_platform import (
    HEF, VDevice, ConfigureParams,
    InputVStreamParams, OutputVStreamParams,
    InferVStreams, HailoStreamInterface, FormatType
)

HEF_PATH = "/home/patriciamolt13/testes/efficientnetlite1.hef"

NUM_IMAGES = 50

class_names = [
    "black_rot",
    "downy_mildew",
    "esca",
    "healthy",
    "powdery_mildew"
]

OUTPUT_DIR = input("Name of the folder: ").strip()

if OUTPUT_DIR == "":
    OUTPUT_DIR = "teste"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 0) Inicializar câmara
# ============================================================

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 1280), "format": "BGR888"}
    )
)

picam2.start_preview(Preview.QT)
picam2.start()

print("Ajusta a folha à frente da câmara...")
print("Serão capturadas", NUM_IMAGES, "imagens.")
time.sleep(10)

# ============================================================
# 1) Inicializar Hailo
# ============================================================

hef = HEF(HEF_PATH)

predictions = []
confidences = []
raw_outputs = []

with VDevice() as target:

    configure_params = ConfigureParams.create_from_hef(
        hef,
        interface=HailoStreamInterface.PCIe
    )

    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_vstreams_params = InputVStreamParams.make(network_group)

    output_vstreams_params = OutputVStreamParams.make(
        network_group,
        format_type=FormatType.FLOAT32
    )

    input_layer_name = hef.get_input_vstream_infos()[0].name
    output_layer_name = hef.get_output_vstream_infos()[0].name

    print("Input layer:", input_layer_name)
    print("Output layer:", output_layer_name)

    with network_group.activate(network_group_params):

        with InferVStreams(
            network_group,
            input_vstreams_params,
            output_vstreams_params
        ) as infer_pipeline:

            for i in range(NUM_IMAGES):

                print(f"\n========== IMAGE {i + 1}/{NUM_IMAGES} ==========")

                # 2) Capturar imagem
                frame = picam2.capture_array()

                # 3) Preparar imagem para o Hailo
                img_240 = cv2.resize(
                    frame,
                    (240, 240),
                    interpolation=cv2.INTER_CUBIC
                )

                img = img_240.astype(np.uint8)
                img = np.expand_dims(img, axis=0)

                print("Input shape:", img.shape)
                print("Input dtype:", img.dtype)
                print("Input min:", img.min())
                print("Input max:", img.max())

                # 4) Inferência
                results = infer_pipeline.infer(
                    {input_layer_name: img}
                )

                raw_output = results[output_layer_name][0]
                raw_output = np.array(raw_output).flatten().astype(np.float32)
                raw_outputs.append(raw_output.copy())

                # Softmax
                exp_scores = np.exp(raw_output - np.max(raw_output))
                probs = exp_scores / np.sum(exp_scores)

                predicted_class = int(np.argmax(probs))
                prediction_name = class_names[predicted_class]
                confidence = probs[predicted_class] * 100

                predictions.append(predicted_class)
                confidences.append(confidence)

                print("Raw output:", raw_output)
                print("Prediction:", prediction_name)
                print("Confidence:", f"{confidence:.2f}%")

                # Guardar apenas a imagem original com o nome da previsão
                predicted_path = os.path.join(
                    OUTPUT_DIR,
                    f"image_{i + 1:02d}_{prediction_name}_{confidence:.1f}.jpg"
                )

                cv2.imwrite(
                    predicted_path,
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )

                if i < NUM_IMAGES - 1:
                    time.sleep(10)

# ============================================================
# TERMINAR CÂMARA
# ============================================================

picam2.stop()
picam2.close()

# ============================================================
# RESUMO
# ============================================================

counts = np.bincount(predictions, minlength=len(class_names))
majority_class = int(np.argmax(counts))

results_txt_path = os.path.join(OUTPUT_DIR, "results.txt")

summary_lines = []

summary_lines.append("=================================")
summary_lines.append("SUMMARY")
summary_lines.append("=================================")

for i, prediction in enumerate(predictions):

    raw_output_str = np.array2string(
        raw_outputs[i],
        precision=4,
        separator=", "
    )

    line = (
        f"Image {i + 1:02d}: "
        f"{class_names[prediction]} "
        f"({confidences[i]:.2f}%)"
        f"\n Raw output: {raw_output_str}"
    )

    summary_lines.append(line)

summary_lines.append("")
summary_lines.append("Predictions per class:")

for class_name, count in zip(class_names, counts):

    percentage = 100.0 * count / len(predictions)

    line = (
        f"{class_name}: "
        f"{count}/{len(predictions)} "
        f"({percentage:.2f}%)"
    )

    summary_lines.append(line)

summary_lines.append("")
summary_lines.append(f"Majority prediction: {class_names[majority_class]}")
summary_lines.append(f"Mean confidence: {np.mean(confidences):.2f}%")
summary_lines.append(f"Saved images: {OUTPUT_DIR}")

# Print summary on terminal
print("\n".join(summary_lines))

# Save summary to txt file
with open(results_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nSaved summary:", results_txt_path)
