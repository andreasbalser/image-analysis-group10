# === 1. Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import string
import time
import os

# === 2. Output Folder Setup ===
# Create folder for saving plots if it doesn't exist
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# === 3. Device Configuration ===
# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 4. Transform Setup ===
# Convert PIL images to PyTorch tensors
transform = transforms.ToTensor()

# === 5. Load EMNIST Dataset (letters split) ===
# EMNIST labels range from 1 to 26 (A to Z)
train_set = EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_set = EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

# === 6. DataLoader Setup ===
# Enables batch loading and shuffling
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000)

# === 7. Label Mapping: Integer (1–26) → Alphabet Letter ===
label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase, start=1)}

# === 8. Plot Sample Images from the Dataset ===
def plot_random_emnist_samples(dataset, num_samples=4):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for ax in axes:
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        # EMNIST images are rotated, so transpose them for correct orientation
        ax.imshow(img.squeeze().T, cmap='gray')
        ax.set_title(label_map[label])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_images.png"))
    plt.show()

plot_random_emnist_samples(train_set)

# === 9. Define a Simple CNN Model ===
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),  # (28x28) → (28x28)
    nn.ReLU(),                                                           # Non-linear activation
    nn.MaxPool2d(kernel_size=2, stride=2),                               # (28x28) → (14x14)
    nn.Flatten(),                                                        # 8×14×14 = 1568
    nn.Linear(in_features=8 * 14 * 14, out_features=26)                  # Map to 26 classes (A–Z)
).to(device)

# === 10. Loss Function and Optimizer ===
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adaptive learning rate

# === 11. Training + Batch Loss Recording ===
def train(model, loader, optimizer, criterion, epoch, loss_list):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        # EMNIST labels go from 1–26 → shift to 0–25
        data, target = data.to(device), (target - 1).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} — Avg. Training Loss: {avg_loss:.4f}")

# === 12. Testing Accuracy Function ===
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), (target - 1).to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# === 13. Training Loop with Timer and Loss Tracking ===
num_epochs = 5
all_batch_losses = []
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    batch_losses = []
    train(model, train_loader, optimizer, criterion, epoch, batch_losses)
    all_batch_losses.append(batch_losses)
    test(model, test_loader)

end_time = time.time()
training_time = end_time - start_time
print(f"\nTotal training time: {training_time:.2f} seconds")

def smooth_losses(losses, factor=20):
    # Groups every `factor` losses and averages them
    return [sum(losses[i:i+factor]) / factor for i in range(0, len(losses), factor)]


# === 14. Plot Loss per Batch ===
plt.figure(figsize=(10, 5))
for epoch_idx, losses in enumerate(all_batch_losses):
    smoothed = smooth_losses(losses, factor=20)  # Smooth out spikes
    plt.plot(smoothed, label=f"Epoch {epoch_idx + 1}")
plt.title("Training Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_loss_plot_smoothed.png"))
plt.show()
