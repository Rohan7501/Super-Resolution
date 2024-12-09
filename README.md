  # Super-Resolution
CSCI 635 - Group 4

# GPU Requirement:
  1. GPU is optional for model training and testing 
  2. GPU is required for converting the Pytorch model to TensorRT engine and running infernce on the engine
     
# Datasets:
  1. DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
  2. BSD300: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
  3. Selfie2Anime*: https://www.kaggle.com/datasets/arnaud58/selfie2anime
  <br />*For Selfie2Anime we just use the anime faces disregarding the human faces

# Training:
Edit the path of training data and testing data folders in "data.py".
<br />Optional: Edit the path where you want to store the models.
<br />You can change the batch size in main.py if your VRAM in insufficient 
<br />Run main.py

# Pre-trained models:
You can access the pre-trained models from this drive link: https://drive.google.com/drive/folders/19WSmRBpaKdZd2yDP8t7_dfwK1ht0UYis?usp=sharing

# Testing/Super resolving an image: 
Run super_resolve.py with following arguments -  
  1. Input image name
  2. Selected model file
  3. Output image name
  4. Cuda option

Example command: python3 super_resolve.py --input_image "Input_Images/1.jpeg" --model "Trained_model/model_epoch_27.pth" --output_filename "Output_Images/output.jpg" --cuda 

Remove --cuda, if you dont have gpu or in other cases, to run on cpu. 

# Converting the model to TensorRT engine 
  Run torch_to_engine.py with following arguments:
  1. model location
  2. output filename(without any extension, the file will be created as "output filename.engine")
  3. Input image height
  4. Input image width

Example command: python3 torch_to_engine.py --model model.pth --output_filename model --input_height 128 --input_width 128

<br />One of the contraints for speeding up the model is we fix the input size for the image.
<br />With our limitation of 8GB vRAM we were able to produce an engine that takes input image of resolution 128x128.
<br />However this constraint doesnt apply for the pytorch model which can handle a 1280x720 image for a 8GB vRam GPU.
<br />The engine file for the pretrained model for 128x128 input size is also available on the above drive link

# Running the engine
Run run_engine.py with following arguments:
1. model location
2. input image
3. output image

Example command: python3 run_engine.py --engine model.engine --input_image 9.jpg --output_filename test.png
