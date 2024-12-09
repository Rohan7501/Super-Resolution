import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch
from time import time

# Load the TensorRT engine
def load_engine(trt_logger, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# # Allocate input and output buffers
def allocate_buffers(engine):
    bindings = []
    input_binding_index = -1
    output_binding_index = -1

    for i in range(engine.num_io_tensors):  # Updated for num_io_tensors
        tensor_name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_binding_index = i
        elif engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
            output_binding_index = i

    # Get input and output tensor shapes
    input_shape = engine.get_tensor_shape(engine.get_tensor_name(input_binding_index))
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(output_binding_index))

    input_dtype = trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(input_binding_index)))
    output_dtype = trt.nptype(engine.get_tensor_dtype(engine.get_tensor_name(output_binding_index)))

    # Convert shape to int to avoid Boost.Python.ArgumentError
    input_size = int(np.prod(input_shape)) * np.dtype(input_dtype).itemsize
    output_size = int(np.prod(output_shape)) * np.dtype(output_dtype).itemsize

    # Allocate device memory for input and output
    input_device = cuda.mem_alloc(input_size)
    output_device = cuda.mem_alloc(output_size)

    # Create host buffers
    input_host = np.empty(input_shape, dtype=input_dtype)
    output_host = np.empty(output_shape, dtype=output_dtype)

    return input_host, input_device, output_host, output_device, input_binding_index, output_binding_index

# Perform inference using the TensorRT engine
def infer(engine, input_tensor):

    # Create execution context
    context = engine.create_execution_context()

    # Allocate buffers
    input_host, input_device, output_host, output_device, input_binding_index, output_binding_index = allocate_buffers(engine)

    # Transfer input data to the device
    cuda.memcpy_htod(input_device, input_tensor.ravel())

    # Set the binding tensors in the context
    context.set_tensor_address(engine.get_tensor_name(input_binding_index), int(input_device))
    context.set_tensor_address(engine.get_tensor_name(output_binding_index), int(output_device))

    # Create a CUDA stream
    stream = cuda.Stream()

    # Execute inference
    if not context.execute_async_v3(stream_handle=stream.handle):
        raise RuntimeError("Inference execution failed.")

    # Transfer the output data back to the host
    cuda.memcpy_dtoh(output_host, output_device)

    return output_host

def main():
    # Set the path to your TensorRT engine file
    engine_path = 'model2.engine'

    # Load the TensorRT engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(trt_logger, engine_path)

    # Load and preprocess the input image
    input_image = 'Input_Images/8.jpg'
    img = Image.open(input_image).convert('RGB')
    img_to_tensor = ToTensor()
    input_tensor = img_to_tensor(img).unsqueeze(0)
    input_tensor = input_tensor.numpy()

    # Perform inference
    output_data = infer(engine, input_tensor)

    # Post-process the output data
    output_tensor = torch.from_numpy(output_data)
    output_tensor = output_tensor.squeeze(0).clamp(0, 1)
    output_img = ToPILImage()(output_tensor)
    output_img.save('Output_Images/test8.png')

    print("Inference completed and output image saved.")

if __name__ == "__main__":
    start = time()
    main()
    print(f"Execution time: {time()-start}")
