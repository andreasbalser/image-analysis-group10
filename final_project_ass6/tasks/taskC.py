import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
import os
import string

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

MODEL_PATH = "saved_models/saved_models_taskB/deepcnn_c116_c232_k5_lr0.001_bs64_ep5_acc94_cpu.pth"

# -----------------------------
# Model Definition (from best experiment)
# -----------------------------
class DeepCNN(nn.Module):
    def __init__(self, channels_1=16, channels_2=32, kernel_size=5, num_classes=26):
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# Load EMNIST Digits Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

data_root = "data"
train_dataset = EMNIST(root=data_root, split="digits", train=True, download=True, transform=transform)
test_dataset = EMNIST(root=data_root, split="digits", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Load Pretrained Model and Modify Last Layer
# -----------------------------
model = DeepCNN(channels_1=16, channels_2=32, kernel_size=5, num_classes=26)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)

# Replace final layer
model.model[-1] = nn.Linear(128, NUM_CLASSES)
model.to(DEVICE)

# Freeze all layers except final layer
for param in model.parameters():
    param.requires_grad = False
for param in model.model[-1].parameters():
    param.requires_grad = True

# -----------------------------
# Training Setup
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.model[-1].parameters(), lr=LEARNING_RATE)

# -----------------------------
# Transfer Learning Training
# -----------------------------
print("Starting transfer learning training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

training_duration = time.time() - start_time
print(f"\nTraining Time: {training_duration:.2f} seconds")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# -----------------------------
# Save Model for Task C
# -----------------------------
def generate_model_name(architecture, channels_1, channels_2, kernel_size, fc_out,
                        num_epochs, optimizer_name, learning_rate, accuracy, device_used, batch_size=None):
    name = (
        f"{architecture}_c1{channels_1}_c2{channels_2}_k{kernel_size}"
        f"_fc{fc_out}_ep{num_epochs}"
        f"_{optimizer_name.lower()}_lr{learning_rate}"
    )
    if batch_size:
        name += f"_bs{batch_size}"
    name += f"_acc{int(round(accuracy))}_{device_used}"
    return name

model_name = generate_model_name(
    architecture="deepcnn_transfer",
    channels_1=16,
    channels_2=32,
    kernel_size=5,
    fc_out=10,
    num_epochs=EPOCHS,
    optimizer_name="Adam",
    learning_rate=LEARNING_RATE,
    accuracy=accuracy,
    device_used=str(DEVICE),
    batch_size=BATCH_SIZE
)

save_dir = "saved_models/saved_models_taskC"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, model_name + ".pth")
torch.save(model.state_dict(), save_path)
print(f"Transfer learning model saved to {save_path}")
