import os

data_dir = "data/chest_xray"

for split in ["train", "val", "test"]:
    normal = len(os.listdir(f"{data_dir}/{split}/NORMAL"))
    pneumonia = len(os.listdir(f"{data_dir}/{split}/PNEUMONIA"))
    total = normal + pneumonia
    print(f"{split:6} → NORMAL: {normal:4d} | PNEUMONIA: {pneumonia:4d} | ratio: {pneumonia/normal:.1f}x")