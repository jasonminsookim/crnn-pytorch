import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import string
from sklearn.preprocessing import LabelEncoder
import xml.etree.ElementTree as ET
import cv2
from PIL import Image as im
import numpy as np
from scipy.ndimage import interpolation as inter


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

        alphanumerics = string.ascii_uppercase + string.digits
        alphanum_arr = []
        for char in alphanumerics:
            alphanum_arr.append(char)
        label_encoder = LabelEncoder()
        label_encoder.fit(alphanum_arr)
        self.label_encoder = label_encoder

    def __len__(self):
        return self.img_meta.shape[0]

    def __getitem__(self, idx):
        img_path = self.img_meta.iloc[idx, 0]
        image = Image.open(img_path)
        label = self.img_meta.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label = [self.CHAR2LABEL[c] for c in label]
        label = torch.LongTensor(label)

        label_length = [len(label)]
        label_length = torch.Tensor(label_length)

        sample_dict = {
            "image": image,
            "label": label,
            "label_length": label_length,
            "idx": idx,
        }
        return sample_dict


def extract_bnd_box(xml_path):
    etree = ET.parse(xml_path)
    root = etree.getroot()
    xmin = 1280
    xmax = 0
    ymin = 720
    ymax = 0

    for box in root.iter("xmin"):
        if float(box.text) < xmin:
            xmin = float(box.text)

    for box in root.iter("xmax"):
        if float(box.text) > xmax:
            xmax = float(box.text)

    for box in root.iter("ymin"):
        if float(box.text) < ymin:
            ymin = float(box.text)

    for box in root.iter("ymax"):
        if float(box.text) > ymax:
            ymax = float(box.text)

    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}


def crop_and_save(file_path, label, xmin, xmax, ymin, ymax, save_dir):
    og_img = Image.open(file_path)
    cropped_img = og_img.crop((xmin, ymin, xmax, ymax))

    cropped_img.save(save_dir / f"{label}.jpg")
    return save_dir / f"{label}.jpg"


def preprocess_cropped_image(cropped_image_path, label, save_dir):
    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    # Reads original image
    img = cv2.imread(cropped_image_path, 0)

    # Applies adaptive threshold image.
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=3, C=3
    )

    # Contour dilates image.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 1))
    img = cv2.dilate(img, kernel)

    # Blurs.
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]

    # Thins image.
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)

    # Fixes skew.
    delta = 0.05
    limit = 4.0
    angles = np.arange(-0.5, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    best_angle = best_angle.round(2)

    img = im.fromarray(img)
    img = img.rotate(best_angle, expand=1, fillcolor="white")
    img.save(save_dir / f"{label}.jpg")
    return save_dir / f"{label}.jpg"
