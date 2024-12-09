from os.path import exists, join, basename
from os import makedirs, remove
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, TenCrop, RandomHorizontalFlip, RandomRotation

from dataset import DatasetFromFolder


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    train_dir = "div2k_data/train"
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    test_dir = "div2k_data/test"
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
