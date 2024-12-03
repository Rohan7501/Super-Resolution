  # Super-Resolution
CSCI 635 - Group 4

# Datasets:
  1. DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
  2. BSD300: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
  3. Selfie2Anime*: https://www.kaggle.com/datasets/arnaud58/selfie2anime
     *For Selfie2Anime we just use the anime faces disregarding the human faces

# Training:
Edit the path of training data and testing data folders in "data.py.
Optional: Edit the path where you want to store the models.
\n Run main.py

# Pre-trained models:
You can access the pre-trained models from this drive link: https://drive.google.com/drive/folders/1fRZq8g9IX6ltdn892OBZY3Le6MOu3SuL?usp=sharing

# Testing/Super resolving an image: 
Run super_resolve.py with following arguments -  
  1. Input image name
  2. Selected model file
  3. Output image name
  4. Cuda option

Example command: python3 super_resolve.py --input_image "Input_Images/1.jpeg" --model "Trained_model/model_epoch_27.pth" --output_filename "Output_Images/output.jpg" --cuda 

Remove --cuda, if you dont have gpu or in other cases, to run on cpu. 
