import torch
import timm
import onnx
import onnxruntime as ort
import numpy as np

# =========================
# CONFIG — must match training script
# =========================
MODEL_NAME = "tf_efficientnet_lite1"
NUM_CLASSES = 5
IMG_SIZE = 240

CHECKPOINT_PATH = "efficientnetlite1.pth"
ONNX_PATH = "efficientnetlite1.onnx"

DEVICE = torch.device("cpu")  


# =========================
# REBUILD MODEL ARCHITECTURE
# =========================
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,      # we're loading your fine-tuned weights, not ImageNet ones
    num_classes=NUM_CLASSES
)

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


# =========================
# DUMMY INPUT
# =========================
# Batch size 1, 3 channels (RGB), IMG_SIZE x IMG_SIZE
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,device=DEVICE)

print(f"Dummy input shape: {tuple(dummy_input.shape)}")

# =========================
# EXPORT TO ONNX
# =========================
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=13,          
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
    dynamo=False
)

print(f"Exported ONNX model to: {ONNX_PATH}")


# =========================
# SANITY CHECK — verify ONNX matches PyTorch output
# =========================

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model is structurally valid.")

ort_session = ort.InferenceSession(ONNX_PATH)

with torch.no_grad():
    torch_output = model(dummy_input).numpy()

onnx_output = ort_session.run(
    None,
    {"input": dummy_input.numpy()}
)[0]

max_diff = np.abs(torch_output - onnx_output).max()
print(f"Max difference between PyTorch and ONNX outputs: {max_diff:.6f}")

if max_diff < 1e-4:
    print("PASS: ONNX output matches PyTorch output closely.")
else:
    print("WARNING: outputs diverge more than expected — check opset/version compatibility.")