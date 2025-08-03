import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 64
EPOCHS = 5  # Transfer learning will train only the final layer
LEARNING_RATE = 0.001  # From Experiment 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  # Digits 0–9

MODEL_PATH = "/Users/mirjamnowotny/Library/Mobile Documents/com~apple~CloudDocs/Uni/Semester 2/Image Analysis/Assignments/image-analysis-group10/ass6/saved_models/saved_models_taskB/deepcnn_64f_fc26_ep10_adam_lr0.001_acc94_cpu.pth"

# -----------------------------
# Define model architecture (same as Task B)
# -----------------------------
class DeepCNN(nn.Module):
    def __init__(self, channels_1=32, channels_2=64, kernel_size=3, num_classes=26):
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
# Load EMNIST 'digits' dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
])

train_dataset = EMNIST(root="./data", split="digits", train=True, download=True, transform=transform)
test_dataset = EMNIST(root="./data", split="digits", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Instantiate model and load pretrained weights
# -----------------------------
model = DeepCNN(channels_1=32, channels_2=64, kernel_size=3, num_classes=26)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)

# -----------------------------
# Replace final layer for 10-digit classification
# -----------------------------
model.model[-1] = nn.Linear(128, NUM_CLASSES)
model.to(DEVICE)

# -----------------------------
# Freeze all layers except final classifier layer
# -----------------------------
for param in model.parameters():
    param.requires_grad = False
for param in model.model[-1].parameters():
    param.requires_grad = True

# -----------------------------
# Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.model[-1].parameters(), lr=LEARNING_RATE)

# -----------------------------
# Transfer Learning Training Loop
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

end_time = time.time()
training_duration = end_time - start_time
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
# Save the trained digit model for Task D
# -----------------------------
save_path = "ass6/saved_models/saved_models_taskC/transfer_digit_model.pth"
torch.save(model.state_dict(), save_path)
print(f"Transfer learning model saved to {save_path}")
