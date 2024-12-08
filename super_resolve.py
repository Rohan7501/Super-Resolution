from __future__ import print_function
import argparse
import torch
from PIL import Image
import torch.nn.functional as F
from math import log10
from torchvision.transforms import ToTensor, ToPILImage
from main import Net, ResidualBlock, EDSR_B, EDSR_F, EDSR_PS, EDSR_scaling_factor, EDSR_scale, SEBlock, SpatialAttention
from time import time

from torchvision.models import vgg19
    
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

start = time()

# Load and preprocess the input image
img = Image.open(opt.input_image).convert('RGB')
print(f"Input image size: {img.size}, mode: {img.mode}")

img_to_tensor = ToTensor()
input_tensor = img_to_tensor(img).unsqueeze(0)
print(f"Input tensor size: {input_tensor.size()}")

print(f"CUDA available: {torch.cuda.is_available()}")


# Determine the device to use
device = torch.device('cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
model = torch.load(opt.model, map_location=device)
model = model.to(device)

# Move the input tensor to the appropriate device
input_tensor = input_tensor.to(device)

model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)

# Post-process the output tensor
output_tensor = output_tensor.cpu().squeeze(0)
output_tensor = output_tensor.clamp(0, 1)  # Ensure the output is in [0, 1] range
print(f"Output tensor size: {output_tensor.size()}")

output_img = ToPILImage()(output_tensor)
output_img.save(opt.output_filename)
print('Output image saved to ', opt.output_filename)

end = time()
print(f"Execution time: {end - start} seconds")