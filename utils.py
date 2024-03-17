import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torchvision.transforms as transforms


class VisDroneSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_seq_len=None):
        """
        Args:
            root_dir (string): Directory with all the sequences and annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_seq_len (int, optional): Maximum sequence length. Sequences will be padded or truncated to this length.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
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

        # Apply transformations
        if self.transform:
            images = [self.transform(image) for image in images]

        # Pad or truncate sequence to max_seq_len
        if self.max_seq_len is not None:
            images = images[: self.max_seq_len]
            pad_len = self.max_seq_len - len(images)
            images += [torch.zeros_like(images[0])] * pad_len

        images = torch.stack(images)

        # Example for loading annotations - adjust according to your needs
        annotation_path = self.annotations[idx]
        with open(annotation_path, "r") as file:
            annotations = file.readlines()

        return images, annotations


def collate_fn(batch):
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
        # Assuming predictions are [x_min, y_min, x_max, y_max]
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
