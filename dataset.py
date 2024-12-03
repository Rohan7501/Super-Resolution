import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    # img = np.array(img, dtype=np.float32)
    # img = Image.open(filepath).convert('YCbCr')
    # y, _, _ = img.split()
    # return y
    # mean = np.array([114.35629928, 111.561547, 103.1545782])
    # img -= mean
    # img = np.clip(img, 0, 255)

    # # Convert back to uint8 (since image data should be in this format)
    # img = img.astype(np.uint8)

    # Convert the NumPy array back to a PIL image
    # img = Image.fromarray(img)
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        # mean = torch.tensor([114.35629928, 111.561547, 103.1545782]) / 255.0
        # img_tensor = img_tensor - mean.view(3, 1, 1)
        if self.input_transform:
            input = self.input_transform(input)
            # input -= mean.view(3, 1, 1)
        if self.target_transform:
            target = self.target_transform(target)
            # target -= mean.view(3, 1, 1)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
