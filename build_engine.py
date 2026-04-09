import tensorrt as trt
import os

os.makedirs("models", exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ONNX_PATH  = "models/resnet50.onnx"
ENGINE_PATH = "models/resnet50_fp16.trt"

print("Building TensorRT FP16 engine... (this takes ~5-15 mins)")

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser  = trt.OnnxParser(network, TRT_LOGGER)

with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
config.set_flag(trt.BuilderFlag.FP16)

profile = builder.create_optimization_profile()
profile.set_shape("input", (1,3,224,224), (8,3,224,224), (16,3,224,224))
config.add_optimization_profile(profile)

print("Compiling...")
serialized = builder.build_serialized_network(network, config)

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)

print(f"Done! Engine saved to {ENGINE_PATH}")