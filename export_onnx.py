import torch
import os
from veritasxr_model import get_model

os.makedirs("models", exist_ok=True)

print("Loading VeritasXR model...")
model = get_model()
model.load_state_dict(torch.load("models/veritasxr.pth"))
model.eval().cuda()

dummy_input = torch.randn(1, 1, 224, 224).cuda()

print("Exporting to ONNX...")
torch.onnx.export(
    model,                          # export full model, no wrapper
    dummy_input,
    "models/veritasxr.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["verdict", "uncertainty"],   # both outputs, correct names
    dynamic_axes={
        "input":       {0: "batch_size"},
        "verdict":     {0: "batch_size"},
        "uncertainty": {0: "batch_size"}
    }
)
print("Done! Saved to models/veritasxr.onnx")