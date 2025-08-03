# === 1. Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
from utils import generate_model_name, save_model, print_and_save_summary, load_emnist_data, plot_random_emnist_samples


# === 2. Output Folder Setup ===
# Create folder for saving plots if it doesn't exist
output_dir = "final_project_ass6/output_images/output_images_taskA"
os.makedirs(output_dir, exist_ok=True)

# === 3. Device Configuration ===
# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 4. Transform Setup ===
# Convert PIL images to PyTorch tensors
transform = transforms.ToTensor()

# Load data
train_loader, test_loader, label_map = load_emnist_data()
train_set = train_loader.dataset  # Needed for plotting

# Plot sample images
sample_image_path = os.path.join(output_dir, "sample_images.png")
plot_random_emnist_samples(train_set, label_map, sample_image_path)


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


final_accuracy = test(model, test_loader)


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

model_name = generate_model_name(
    architecture="cnn1",  # or "deepcnn"
    num_filters=8,        # or 32
    fc_out=26,
    num_epochs=num_epochs,
    optimizer_name="Adam",
    learning_rate=0.001,
    accuracy=final_accuracy,
    device_used=str(device)
)

model_path = save_model(model, model_name, output_dir="final_project_ass6/saved_models/saved_models_taskA")  # or taskB

print_and_save_summary(
    device=device,
    num_epochs=num_epochs,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer_name="Adam",
    learning_rate=0.001,
    training_time=training_time,
    loss_plot_path=os.path.join(output_dir, "training_loss_plot_smoothed.png"),
    sample_plot_path=os.path.join(output_dir, "sample_images.png"),
    final_accuracy=final_accuracy,
    model_path=model_path,
    output_dir=output_dir
)
