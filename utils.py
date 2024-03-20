import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torchvision.transforms as transforms


class VisDroneSequenceDataset(Dataset):
    """
    This class is used to load and process the VisDrone dataset.

    Args:
        root_dir (str): The root directory of the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        root_dir (str): The root directory of the dataset.
        transform (callable): A function/transform that takes in an PIL image and returns a transformed version.
        sequence_dirs (list): A list of directories containing the image sequences.
        annotations (list): A list of paths to the annotation files.

    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Returns the images and annotations for the sequence at the given index.
        collate_fn(batch): Collates a batch of data.
        load_annotations(annotation_path): Loads the annotations from the given file.
        visualize_predictions(images, predictions): Visualizes the given images and predictions.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_dirs = [
            os.path.join(root_dir, "sequences", d)
            for d in os.listdir(os.path.join(root_dir, "sequences"))
        ]
        self.annotations = [
            os.path.join(root_dir, "annotations", f)
            for f in os.listdir(os.path.join(root_dir, "annotations"))
            if f.endswith("_v.txt")
        ]

    def __len__(self):
        return len(self.sequence_dirs)

    def load_annotations(self, annotation_path):
        """
        Loads annotations from a text file and returns a list of bounding boxes.
        The expected format of each line in the text file is:
        "x1,y1,x2,y2,label"
        """
        annotations = []
        with open(annotation_path, "r") as file:
            for line in file:
                values = line.strip().split(",")
                x1, y1, x2, y2 = map(
                    float, values[:4]
                )  # Convert the first 4 values to float
                label = int(values[4])  # Convert the last value to int
                annotations.append([x1, y1, x2, y2, label])
        return annotations

    def __getitem__(self, idx):
        sequence_dir = self.sequence_dirs[idx]
        image_files = sorted(
            [
                os.path.join(sequence_dir, f)
                for f in os.listdir(sequence_dir)
                if f.endswith(".jpg")
            ]
        )
        images = [Image.open(f).convert("RGB") for f in image_files]

        # Load and format annotations
        annotations = self.load_annotations(self.annotations[idx])

        if self.transform:
            images = [self.transform(image) for image in images]

        images = torch.stack(images, dim=0)  # Shape: (seq_len, channels, height, width)
        annotations = torch.tensor(annotations)  # Ensure correct format

        return images, annotations

    def collate_fn(self, batch):
        images = [item[0] for item in batch]  # List of lists
        annotations = [item[1] for item in batch]
        return images, annotations

    def visualize_predictions(images, predictions):
        """
        Visualizes sequences of images with bounding box predictions.
        Args:
            - images (list of PIL Images): The sequence of images.
            - predictions (Tensor): The bounding box predictions for the sequence.
        """
        fig, axs = plt.subplots(len(images), 1, figsize=(5, len(images) * 5))
        if len(images) == 1:  # If there's only one image, axs is not a list
            axs = [axs]

        for i, image in enumerate(images):
            axs[i].imshow(image)
            # Assuming predictions are [x_min, y_min, x_max, y_max, label]
            pred = predictions[i].detach().cpu().numpy()
            rect = patches.Rectangle(
                (pred[0], pred[1]),
                pred[2] - pred[0],
                pred[3] - pred[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            axs[i].add_patch(rect)

        plt.show()
