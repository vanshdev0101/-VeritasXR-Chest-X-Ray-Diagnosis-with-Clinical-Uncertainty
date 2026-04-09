import torch
import torch.nn as nn
import torchvision.models as models
import os

os.makedirs("models", exist_ok=True)

print("Loading fine-tuned chest X-ray model...")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet50_xray.pth"))
model.eval().cuda()

dummy_input = torch.randn(1, 3, 224, 224).cuda()

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "models/resnet50.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("Done! Saved to models/resnet50.onnx")