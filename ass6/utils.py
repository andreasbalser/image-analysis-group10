# === utils.py ===

import os
import torch
import string
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader


# === Data Loading ===
def load_emnist_data(root: str = "ass6/data", batch_size_train: int = 64, batch_size_test: int = 1000):
    """
    Load EMNIST (letters split) and return DataLoaders for train and test sets,
    along with a label map.
    """
    transform = transforms.ToTensor()

    train_set = EMNIST(root=root, split='letters', train=True, download=True, transform=transform)
    test_set = EMNIST(root=root, split='letters', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test)

    label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase, start=1)}

    return train_loader, test_loader, label_map


# === Sample Plotting ===
def plot_random_emnist_samples(dataset, label_map, output_path, num_samples=4):
    """
    Plot and save a few random samples from the EMNIST dataset with their labels.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for ax in axes:
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        ax.imshow(img.squeeze().T, cmap='gray')  # Transpose due to EMNIST format
        ax.set_title(label_map[label])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# === CNN Model Definition ===
class DeepCNN(nn.Module):
    def __init__(self, channels_1=16, channels_2=32, kernel_size=3):
        """
        A configurable CNN with two convolutional layers,
        followed by fully connected layers.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, channels_1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(channels_1, channels_2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels_2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 26)  # 26 classes (A-Z)
        )

    def forward(self, x):
        return self.model(x)


# === Model Naming Utility ===
def generate_model_name(architecture: str, num_filters: int, fc_out: int,
                        num_epochs: int, optimizer_name: str,
                        learning_rate: float, accuracy: float, device_used: str) -> str:
    """
    Generate a standardized filename for saving the model.
    """
    return (f"{architecture}_{num_filters}f_fc{fc_out}_ep{num_epochs}_"
            f"{optimizer_name.lower()}_lr{learning_rate}_acc{int(round(accuracy))}_{device_used}")


# === Model Saving ===
def save_model(model, model_name: str, output_dir: str = "saved_models") -> str:
    """
    Save the model to the specified directory and return its path.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, model_name + ".pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    return save_path


# === Training Summary Logger ===
def print_and_save_summary(device,
                           num_epochs,
                           train_loader,
                           test_loader,
                           optimizer_name,
                           learning_rate,
                           training_time,
                           loss_plot_path,
                           sample_plot_path,
                           final_accuracy,
                           model_path,
                           output_dir="output_images") -> None:
    """
    Print and save the training summary to a text file.
    """
    summary = (
        "\n=== TRAINING SUMMARY ===\n"
        f"Device used: {device}\n"
        f"Number of epochs: {num_epochs}\n"
        f"Training batch size: {train_loader.batch_size}\n"
        f"Testing batch size: {test_loader.batch_size}\n"
        f"Optimizer: {optimizer_name}\n"
        f"Learning rate: {learning_rate}\n"
        f"Total training time: {training_time:.2f} seconds\n"
        f"Loss plot saved to: {loss_plot_path}\n"
        f"Sample image plot saved to: {sample_plot_path}\n"
        f"Final Test Accuracy: {final_accuracy:.2f}%\n"
        f"Model saved to: {model_path}\n"
    )

    print(summary)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
