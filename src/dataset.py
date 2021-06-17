import os
import glob

import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import string
import pandas as pd


class Synth90kDataset(Dataset):
    CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(
        self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100
    ):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, "lexicon.txt"), "r") as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == "train":
            paths_file = "annotation_train.txt"
        elif mode == "dev":
            paths_file = "annotation_val.txt"
        elif mode == "test":
            paths_file = "annotation_test.txt"

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), "r") as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(" ")
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert("L")  # grey-scale
        except IOError:
            print("Corrupted image for %d" % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


def extract_jpg_meta(img_dir, img_type):
    # TODO: Create documentation
    # Extracts the file paths that match the .jpg extension.
    jpg_files = img_dir.glob("*.jpg")
    jpg_path_arr = [jpg_file for jpg_file in jpg_files]

    # Extracts the label for the image.
    if img_type == "wbsin":
        y_arr = [
            jpg_path.parts[-1].split("_")[-1].split(".jpg")[0]
            for jpg_path in jpg_path_arr
        ]

    # Creates a data frame for the file path and image labels.
    meta_df = (
        pd.DataFrame([jpg_path_arr, y_arr])
        .transpose()
        .rename(columns={0: "file_path", 1: "label"})
    )

    # Adds dimension columns to the meta_df
    meta_df["width"] = meta_df["file_path"].apply(lambda x: Image.open(x).size[0])
    meta_df["height"] = meta_df["file_path"].apply(lambda x: Image.open(x).size[1])
    return meta_df


class WbsinImageDataset(Dataset):
    CHARS = "XCE9PM8L167AK20F43BD5"
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, meta_file, transform=None, target_transform=None):
        self.img_meta = pd.read_csv(meta_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img_meta.shape[0]

    def __getitem__(self, idx):
        img_path = self.img_meta.iloc[idx, 10]
        image = Image.open(img_path)
        label = self.img_meta.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label = [self.CHAR2LABEL[c] for c in label]
        label = torch.LongTensor(label)

        label_length = [len(label)]
        label_length = torch.LongTensor(label_length)

        sample_dict = {
            "image": image,
            "label": label,
            "label_length": label_length,
            "idx": idx,
        }
        return image, label, label_length


def wbsin_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

