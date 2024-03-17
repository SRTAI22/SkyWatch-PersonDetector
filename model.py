import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.nn.utils import parametrizations
import copy
from utils import VisDroneSequenceDataset, collate_fn


# define gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
num_epochs = 10
learning_rate = 0.005
max_seq_len = 100

# transformations
transform = Compose(
    [
        Resize((800, 800)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# CNN-TCN Model
class TCNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2
    ):
        super(TCNBlock, self).__init__()
        self.conv1 = parametrizations.weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            )
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.conv1(x)))


class CNN_TCN(nn.Module):
    def __init__(self, max_seq_len=None):
        super(CNN_TCN, self).__init__()
        self.max_seq_len = max_seq_len
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final FC layer
        self.tcn = nn.Sequential(
            TCNBlock(512, 256),  # Match input size with CNN output size
            TCNBlock(256, 128),
            TCNBlock(128, 64),
            TCNBlock(64, 32),
            TCNBlock(32, 16),
            TCNBlock(16, 8),
            TCNBlock(8, 4),
        )
        self.classifier = nn.Linear(4, 4)  # Final classifier

    def forward(self, x):
        if isinstance(x, list):
            print("List of tensors")  # Debugging
            x = torch.stack(x)
        # shape (C, H, W)
        batch_size = len(x)
        seq_len = self.max_seq_len

        cnn_output = torch.zeros(
            batch_size, seq_len, 512, device=device
        )  # Prepare tensor to store CNN outputs

        # Iterate through each sequence and each image within the sequence
        for i in range(seq_len):
            # Process each frame through CNN
            frame_output = self.cnn(x[:, i, :, :, :])  # shape: [batch_size, C, H, W]
            cnn_output[:, i, :] = (
                frame_output.squeeze()
            )  # Squeeze to remove unnecessary dimensions

        cnn_out = torch.stack(
            cnn_output, dim=1
        )  # Stack the CNN outputs along the sequence dimension

        # Reshape for TCN
        cnn_out = cnn_out.permute(
            0, 2, 1
        )  # Change to (batch_size, features, seq_len) for TCN

        tcn_out = self.tcn(cnn_out)

        tcn_out = tcn_out[:, :, -1]  # Take the output from the last time step

        out = self.classifier(tcn_out)

        return out


# model
model = CNN_TCN(max_seq_len=max_seq_len).to(device)

# Loss and optimiser
mse = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for images, annotations in train_loader:
            images = [img.to(device) for img in images]  # List of tensors

            optimizer.zero_grad()

            outputs = model(images)  # Pass the list of tensors directly
            loss = 0
            for output, annotation in zip(outputs, annotations):
                bboxes = []
                bbox_strs = annotation[0].split(",")
                for bbox_str in bbox_strs:
                    if bbox_str:
                        bbox_values = list(map(float, bbox_str.split(",")))
                        bbox_values = bbox_values + [0.0] * (4 - len(bbox_values))
                        x1, y1, x2, y2 = bbox_values
                        bbox = torch.tensor([x1, y1, x2, y2], device=device)
                        bboxes.append(bbox)
                labels = torch.stack(bboxes)
                loss += criterion(output, labels)
            loss /= len(annotations)  # Normalise loss by batch size
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, annotations in val_loader:
                images = [img.to(device) for img in images]
                labels = []
                for annotation in annotations:
                    bboxes = []
                    bbox_strs = annotation[0].split(",")
                    for bbox_str in bbox_strs:
                        if bbox_str:
                            bbox_values = map(float, bbox_str.split(","))
                            bbox_values = list(bbox_values) + [0.0] * (
                                4 - len(bbox_values)
                            )  # Pad with zeros
                            x1, y1, x2, y2 = bbox_values
                            bbox = torch.tensor([x1, y1, x2, y2], device=device)
                            bboxes.append(bbox)
                    labels.append(bboxes)

                outputs = [model([img]) for img in images]
                loss = 0
                for output, label in zip(outputs, labels):
                    for bbox in label:
                        loss += criterion(output, bbox)
                loss /= len(labels)  # Normalise loss by batch size

                val_loss += loss.item() * len(images)

        val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoch {epoch}/{num_epochs - 1}, Training Loss: {epoch_loss}, Validation Loss: {val_loss}"
        )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping")
                break

    model.load_state_dict(best_model_wts)
    return model


# test model
def test_model(model, test_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    return test_loss


# load data
dataset_train = VisDroneSequenceDataset(
    root_dir="../data/VisDrone2019-VID-train",
    transform=transform,
    max_seq_len=max_seq_len,
)

dataset_val = VisDroneSequenceDataset(
    root_dir="../data/VisDrone2019-VID-val",
    transform=transform,
    max_seq_len=max_seq_len,
)

dataset_test = VisDroneSequenceDataset(
    root_dir="../data/VisDrone2019-VID-test-dev",
    transform=transform,
    max_seq_len=max_seq_len,
)

# data loader
train_loader = DataLoader(
    dataset_train, batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(dataset_val, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(
    dataset_test, batch_size=4, shuffle=True, collate_fn=collate_fn
)


# train model
model = train_model(
    model, train_loader, val_loader, mse, optimiser, num_epochs, patience=5
)

# test model
test_loss = test_model(model, test_loader, mse)

print(f"Test Loss: {test_loss}")


# save model
torch.save(model.state_dict(), "model.pth")
