# Super-Resolution
CSCI 635 - Group 4

# Training:
Edit the path of training data and testing data folders in "data.py.
Run main.py

# Testing/Super resolving an image: 
Run super_resolve.py with following arguments -  
  1. Input image name
  2. Selected model file
  3. Output image name
  4. Cuda option

Example command: python3 super_resolve.py --input_image "Input_Images/1.jpeg" --model "Trained_model/model_epoch_27.pth" --output_filename "Output_Images/output.jpg" --cuda 

Remove --cuda, if you dont have gpu or in other cases, to run on cpu. 
