import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import os
import shutil
import string
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader

output_dir_base = "output_images/output_images_taskB"
model_dir_test = "saved_models/test_models"
model_dir_final = "saved_models/saved_models_taskB"
data_dir = "data"
os.makedirs(output_dir_base, exist_ok=True)
os.makedirs(model_dir_test, exist_ok=True)
os.makedirs(model_dir_final, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_emnist_data(batch_size_train=64, batch_size_test=1000):
    transform = transforms.ToTensor()
    train_set = EMNIST(root=data_dir, split='letters', train=True, download=True, transform=transform)
    test_set = EMNIST(root=data_dir, split='letters', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test)
    label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase, start=1)}
    return train_loader, test_loader, label_map

# deep model
class DeepCNN(nn.Module):
    def __init__(self, channels_1=16, channels_2=32, kernel_size=3):
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
            nn.Linear(128, 26)
        )

    def forward(self, x):
        return self.model(x)
    
# helping fcts
def generate_model_name(arch, lr, batch_size, epochs, acc, device):
    return (f"deepcnn_c1{arch['channels_1']}_c2{arch['channels_2']}_k{arch['kernel_size']}"
            f"_lr{lr}_bs{batch_size}_ep{epochs}_acc{int(round(acc))}_{device}")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")

def smooth_losses(losses, factor=20):
    return [sum(losses[i:i+factor]) / factor for i in range(0, len(losses), factor)]

def test_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), (target - 1).to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100. * correct / total

def print_and_save_summary(path, content):
    with open(path, "a") as f:
        f.write(content)
    print(content)

# different architectures and parameters
architectures = [
    {"channels_1": 8, "channels_2": 16, "kernel_size": 3},
    {"channels_1": 16, "channels_2": 32, "kernel_size": 3},
    {"channels_1": 32, "channels_2": 64, "kernel_size": 3},
    {"channels_1": 16, "channels_2": 32, "kernel_size": 5},
]

hyperparams = [
    {"lr": 0.001, "batch_size": 64, "epochs": 5},
    {"lr": 0.0005, "batch_size": 64, "epochs": 5},
    {"lr": 0.001, "batch_size": 128, "epochs": 5},
    {"lr": 0.001, "batch_size": 64, "epochs": 10},
]

summary_file = os.path.join(output_dir_base, "combined_experiment_summary.txt")
open(summary_file, 'w').write("=== COMBINED EXPERIMENT SUMMARY (Task B) ===\n")

# run experiments
best_accuracy = 0.0
best_model_path = ""
experiment_id = 1
results = []

for arch in architectures:
    for hparam in hyperparams:
        print(f"\n=== Experiment {experiment_id} ===")

        train_loader, test_loader, _ = load_emnist_data(
            batch_size_train=hparam['batch_size'], batch_size_test=1000)

        model = DeepCNN(**arch).to(device)
        optimizer = optim.Adam(model.parameters(), lr=hparam['lr'])
        criterion = nn.CrossEntropyLoss()

        all_batch_losses = []
        start_time = time.time()

        for epoch in range(1, hparam['epochs'] + 1):
            batch_losses = []
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), (target - 1).to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            all_batch_losses.append(batch_losses)
            acc = test_acc(model, test_loader)
            print(f"Epoch {epoch}: Accuracy = {acc:.2f}%")

        training_time = time.time() - start_time
        final_acc = test_acc(model, test_loader)

        # Save 
        model_name = generate_model_name(arch, hparam['lr'], hparam['batch_size'], hparam['epochs'], final_acc, str(device))
        model_path = os.path.join(model_dir_test, model_name + ".pth")
        save_model(model, model_path)

        # best model
        if final_acc > best_accuracy:
            best_accuracy = final_acc
            best_model_path = model_path

        # Save experiment summary
        summary = (
            f"\n--- EXPERIMENT {experiment_id} ---\n"
            f"Channels: {arch['channels_1']} → {arch['channels_2']}, Kernel: {arch['kernel_size']}\n"
            f"Learning Rate: {hparam['lr']}, Batch Size: {hparam['batch_size']}, Epochs: {hparam['epochs']}\n"
            f"Final Accuracy: {final_acc:.2f}%\n"
            f"Training Time: {training_time:.2f} sec\n"
            f"Model Path: {model_path}\n"
        )
        print_and_save_summary(summary_file, summary)
        results.append((experiment_id, final_acc))
        experiment_id += 1
# put best model in correct folder
if best_model_path:
    final_best_model_path = os.path.join(model_dir_final, os.path.basename(best_model_path))
    shutil.copy(best_model_path, final_best_model_path)
    print(f"\nBest model copied to: {final_best_model_path}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")

print("\n=== Experiment Summary ===")
for eid, acc in results:
    print(f"Experiment {eid}: Accuracy = {acc:.2f}%")
print(f"\nBest Accuracy: {best_accuracy:.2f}%")
