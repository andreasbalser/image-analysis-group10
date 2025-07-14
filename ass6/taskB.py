# === taskB.py ===

# === 1. Imports ===
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
from utils import generate_model_name, save_model, print_and_save_summary, load_emnist_data

# === 2. Setup Output Folder ===
output_dir = "ass6/output_images/output_images_taskB"
os.makedirs(output_dir, exist_ok=True)

# === 3. Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 4. Data Setup ===
train_loader, test_loader, label_map = load_emnist_data()

# === 6. Define a Deeper CNN Model ===
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 26)
        )
    def forward(self, x):
        return self.model(x)

model = DeepCNN().to(device)

# === 7. Training Setup ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

# === 8. Training and Testing ===
def train(model, loader, optimizer, criterion, epoch, loss_list):
    model.train()
    total_loss = 0
    for data, target in loader:
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
    return accuracy

# === 9. Training Loop ===
all_batch_losses = []
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    batch_losses = []
    train(model, train_loader, optimizer, criterion, epoch, batch_losses)
    all_batch_losses.append(batch_losses)
    acc = test(model, test_loader)
    print(f"Epoch {epoch} Accuracy: {acc:.2f}%")

end_time = time.time()
training_time = end_time - start_time
final_accuracy = test(model, test_loader)
print(f"Final Test Accuracy: {final_accuracy:.2f}%")
print(f"Total training time: {training_time:.2f} seconds")

# === 10. Plot Loss ===
def smooth_losses(losses, factor=20):
    return [sum(losses[i:i+factor]) / factor for i in range(0, len(losses), factor)]

plt.figure(figsize=(10, 5))
for epoch_idx, losses in enumerate(all_batch_losses):
    smoothed = smooth_losses(losses)
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

model_path = save_model(model, model_name, output_dir="ass6/saved_models/saved_models_taskB")  # or taskB

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
