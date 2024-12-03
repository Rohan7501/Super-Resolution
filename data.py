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
        # TenCrop(crop_size),
        # RandomHorizontalFlip(p=0.5),
        # RandomRotation(degrees=(0,90)),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        # TenCrop(crop_size),
        Resize(crop_size),
        ToTensor(),
    ])


def get_training_set(upscale_factor):
    # train_dir = "/home/rohan/RIT/Summer Sem 2238/CSCI - 631 CV/Project/Code/DIV2K_train_HR"
    train_dir = "/home/rohan/RIT/Summer Sem 2238/CSCI - 631 CV/Project/Code/Combined_dataset_train"
    crop_size = calculate_valid_crop_size(128, upscale_factor)
    # crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


def get_test_set(upscale_factor):
    # test_dir = "/home/rohan/RIT/Summer Sem 2238/CSCI - 631 CV/Project/Code/DIV2K_valid_HR"
    test_dir = "/home/rohan/RIT/Summer Sem 2238/CSCI - 631 CV/Project/Code/Combined_dataset_test"
    crop_size = calculate_valid_crop_size(128, upscale_factor)
    # crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
