import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.nn.utils import parametrizations
import copy
from utils import VisDroneSequenceDataset


# define gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
num_epochs = 10
learning_rate = 0.005

# transformations
transform = Compose(
    [
        Resize((800, 800)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# CNN-LTSM Model
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final FC layer
        self.lstm = nn.LSTM(
            512, 256, batch_first=True
        )  # Match input size with CNN output size
        self.classifier = nn.Linear(256, 4)  # Final classifier

    def forward(self, x):
        print(f"x shape:\n{x[0].shape}")
        # Iterate through each sequence in the batch
        outputs = []
        for seq in x:
            cnn_output = torch.zeros(
                seq.shape[0], 512, device=device
            )  # Per-sequence CNN output

            # Iterate through each frame in the sequence
            for i in range(seq.shape[0]):
                frame = seq[i].unsqueeze(
                    0
                )  # Add a batch dimension for the single frame
                # Ensure the frame has 3 channels
                if frame.shape[1] != 3:
                    frame = frame.repeat(1, 3, 1, 1)
                print(f"frame shape:\n{frame.shape}")
                frame_output = self.cnn(frame)
                cnn_output[i, :] = frame_output.squeeze()

            # Reshape and pass through LSTM
            cnn_out = cnn_output.unsqueeze(0)  # Add a batch dimension
            lstm_out, _ = self.lstm(cnn_out)
            out = self.classifier(lstm_out[-1, :, :])  # Output from last time step

            outputs.append(out)
        return torch.stack(outputs)


# model
model = CNN_LSTM().to(device)

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
            images = [
                img.to(device) for img_seq in images for img in img_seq
            ]  # Send each tensor to the device

            optimizer.zero_grad()

            outputs = model(images)  # Forward pass
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
                images = [
                    img.to(device) for img_seq in images for img in img_seq
                ]  # Send each tensor to the device
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

                outputs = [model(img_seq) for img_seq in images]
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
)

dataset_val = VisDroneSequenceDataset(
    root_dir="../data/VisDrone2019-VID-val",
    transform=transform,
)

dataset_test = VisDroneSequenceDataset(
    root_dir="../data/VisDrone2019-VID-test-dev",
    transform=transform,
)

# data loader
train_loader = DataLoader(
    dataset_train, batch_size=4, shuffle=False, collate_fn=dataset_train.collate_fn
)
val_loader = DataLoader(
    dataset_val, batch_size=4, shuffle=False, collate_fn=dataset_val.collate_fn
)
test_loader = DataLoader(
    dataset_test, batch_size=4, shuffle=False, collate_fn=dataset_test.collate_fn
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
