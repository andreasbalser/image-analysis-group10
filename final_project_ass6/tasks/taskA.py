# === 1. Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
import string
import random
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader

# === 2. Output Folder Setup ===
output_dir = "output_images/output_images_taskA"
os.makedirs(output_dir, exist_ok=True)

# === 3. Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 4. Data Loading ===
transform = transforms.ToTensor()
train_set = EMNIST(root="data", split='letters', train=True, download=True, transform=transform)
test_set = EMNIST(root="data", split='letters', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000)

label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase, start=1)}

# === 5. Plot Random Sample Images ===
def plot_random_emnist_samples(dataset, label_map, output_path, num_samples=4):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for ax in axes:
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        ax.imshow(img.squeeze().T, cmap='gray')
        ax.set_title(label_map[label])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

sample_image_path = os.path.join(output_dir, "sample_images.png")
plot_random_emnist_samples(train_set, label_map, sample_image_path)

# === 6. Define Simple CNN Model (Figure 2) ===
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=8 * 14 * 14, out_features=26)
).to(device)

# === 7. Loss Function and Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 8. Training and Testing Functions ===
def train(model, loader, optimizer, criterion, epoch, loss_list):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
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

# === 9. Training Loop ===
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

# Final test
final_accuracy = test(model, test_loader)

# === 10. Plot Loss ===
def smooth_losses(losses, factor=20):
    return [sum(losses[i:i+factor]) / factor for i in range(0, len(losses), factor)]

plt.figure(figsize=(10, 5))
for epoch_idx, losses in enumerate(all_batch_losses):
    smoothed = smooth_losses(losses, factor=20)
    plt.plot(smoothed, label=f"Epoch {epoch_idx + 1}")
plt.title("Training Loss per Batch")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_path = os.path.join(output_dir, "training_loss_plot_smoothed.png")
plt.savefig(loss_plot_path)
plt.show()

# === 11. Save Model ===
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")

model_path = f"saved_models/saved_models_taskA/cnn1_8f_fc26_ep{num_epochs}_adam_lr0.001_acc{int(final_accuracy)}_{str(device)}.pth"
save_model(model, model_path)

# === 12. Summary Logging ===
def print_and_save_summary(summary_path):
    summary = (
        "\n=== TRAINING SUMMARY ===\n"
        f"Device used: {device}\n"
        f"Number of epochs: {num_epochs}\n"
        f"Training batch size: {train_loader.batch_size}\n"
        f"Testing batch size: {test_loader.batch_size}\n"
        f"Optimizer: Adam\n"
        f"Learning rate: 0.001\n"
        f"Total training time: {training_time:.2f} seconds\n"
        f"Loss plot saved to: {loss_plot_path}\n"
        f"Sample image plot saved to: {sample_image_path}\n"
        f"Final Test Accuracy: {final_accuracy:.2f}%\n"
        f"Model saved to: {model_path}\n"
    )

    print(summary)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(summary)

summary_path = os.path.join(output_dir, "training_summary.txt")
print_and_save_summary(summary_path)