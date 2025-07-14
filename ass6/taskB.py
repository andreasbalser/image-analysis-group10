# === taskB.py ===

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
from utils import (
    generate_model_name,
    save_model,
    print_and_save_summary,
    load_emnist_data,
    plot_random_emnist_samples,
    DeepCNN
)

# === 1. Setup ===
output_dir_base = "ass6/output_images/output_images_taskB"
model_dir_base = "ass6/saved_models/saved_models_taskB"
summary_file_path = os.path.join(output_dir_base, "experiment_summary.txt")

os.makedirs(output_dir_base, exist_ok=True)
os.makedirs(model_dir_base, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 2. Load Data ===
train_loader, test_loader, label_map = load_emnist_data()
train_set = train_loader.dataset
plot_random_emnist_samples(train_set, label_map, os.path.join(output_dir_base, "sample_images.png"))

# === 3. Training/Test Functions ===
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
    return 100. * correct / total

def smooth_losses(losses, factor=20):
    return [sum(losses[i:i+factor]) / factor for i in range(0, len(losses), factor)]

# === 4. Run Multiple Experiments ===
configs = [
    {"channels_1": 8, "channels_2": 16, "kernel_size": 3},
    {"channels_1": 16, "channels_2": 32, "kernel_size": 3},
    {"channels_1": 32, "channels_2": 64, "kernel_size": 3},
    {"channels_1": 16, "channels_2": 32, "kernel_size": 5},
]

num_epochs = 5
learning_rate = 0.001
optimizer_name = "Adam"

with open(summary_file_path, "w") as summary_file:
    summary_file.write("=== EXPERIMENT SUMMARY (Task B) ===\n")

    for idx, cfg in enumerate(configs):
        print(f"\n=== Running Config {idx+1}: {cfg} ===")
        model = DeepCNN(**cfg).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

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

        # === Save plot ===
        plt.figure(figsize=(10, 5))
        for epoch_idx, losses in enumerate(all_batch_losses):
            smoothed = smooth_losses(losses)
            plt.plot(smoothed, label=f"Epoch {epoch_idx + 1}")
        plt.title(f"Loss – Config {idx+1}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_plot_path = os.path.join(output_dir_base, f"loss_config{idx+1}.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # === Save model ===
        model_name = generate_model_name(
            architecture="deepcnn",
            num_filters=cfg["channels_2"],
            fc_out=26,
            num_epochs=num_epochs,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            accuracy=final_accuracy,
            device_used=str(device)
        )
        model_path = save_model(model, model_name, output_dir=model_dir_base)

        # === Write individual summary ===
        print_and_save_summary(
            device=device,
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            training_time=training_time,
            loss_plot_path=loss_plot_path,
            sample_plot_path=os.path.join(output_dir_base, "sample_images.png"),
            final_accuracy=final_accuracy,
            model_path=model_path,
            output_dir=output_dir_base
        )

        # === Write to combined summary file ===
        summary_file.write(f"\n--- CONFIG {idx+1} ---\n")
        summary_file.write(f"Channels: {cfg['channels_1']} → {cfg['channels_2']}, Kernel Size: {cfg['kernel_size']}\n")
        summary_file.write(f"Final Accuracy: {final_accuracy:.2f}%\n")
        summary_file.write(f"Training Time: {training_time:.2f} sec\n")
        summary_file.write(f"Model Path: {model_path}\n")
        summary_file.write(f"Loss Plot: {loss_plot_path}\n")
