import torch
import tensorrt as trt
import numpy as np
import argparse
from torch.autograd import Variable
from main import Net, ResidualBlock, EDSR_B, EDSR_F, EDSR_PS, EDSR_scaling_factor, EDSR_scale, SpatialAttention,SEBlock

parser = argparse.ArgumentParser(description='Convert pytoch model TensorRT Engine')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str,required=True, help='where to save the output image')
parser.add_argument('--input_height', type=int,required=True, help='input image height')
parser.add_argument('--input_width', type=int,required=True, help='input image width')

opt = parser.parse_args()

# Paths
# pytorch_model_path = "models/model_epoch_27.pth"  # Path to the PyTorch model
pytorch_model_path = opt.model# Path to the PyTorch model
onnx_model_path = opt.output_filename + ".onnx" # Path to save ONNX model
engine_output_path = opt.output_filename + ".engine"  # Path to save TensorRT engine

# 1. Load PyTorch Model
print("Loading PyTorch model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(pytorch_model_path, map_location=device)
model.eval()

# 2. Export Model to ONNX
print("Exporting PyTorch model to ONNX...")
dummy_input = torch.randn(1, 3, opt.input_height, opt.input_width, device=device)  # Adjust input shape as per your model
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=11,  # Ensure compatibility
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)
print(f"ONNX model saved at: {onnx_model_path}")

# 3. Convert ONNX to TensorRT Engine
print("Converting ONNX model to TensorRT engine...")
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX file
with open(onnx_model_path, "rb") as onnx_file:
    if not parser.parse(onnx_file.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX model.")

# Build TensorRT engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # Set workspace size to 1GB
if torch.cuda.is_available():
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 if GPU supports it

engine = builder.build_serialized_network(network, config)
if not engine:
    raise RuntimeError("Failed to build the TensorRT engine.")

# Save TensorRT engine
with open(engine_output_path, "wb") as f:
    f.write(engine)
print(f"TensorRT engine saved at: {engine_output_path}")
