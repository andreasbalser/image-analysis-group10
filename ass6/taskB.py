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

# === 1. Setup Directories ===
output_dir_base = "ass6/output_images/output_images_taskB"
model_dir_base = "ass6/saved_models/saved_models_taskB"
summary_file_path = os.path.join(output_dir_base, "combined_experiment_summary.txt")
os.makedirs(output_dir_base, exist_ok=True)
os.makedirs(model_dir_base, exist_ok=True)

# === 2. Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 3. Initial Data Preview ===
# Load EMNIST once to preview some samples
train_loader_preview, _, label_map = load_emnist_data(batch_size_train=64)
plot_random_emnist_samples(train_loader_preview.dataset, label_map, os.path.join(output_dir_base, "sample_images.png"))

# === 4. Define experiment variations ===

# --- Task C: Vary model architecture ---
architectures = [
    {"channels_1": 8, "channels_2": 16, "kernel_size": 3},
    {"channels_1": 16, "channels_2": 32, "kernel_size": 3},
    {"channels_1": 32, "channels_2": 64, "kernel_size": 3},
    {"channels_1": 16, "channels_2": 32, "kernel_size": 5},
]

# --- Task D: Vary training hyperparameters ---
hyperparams = [
    {"lr": 0.001, "batch_size": 64, "epochs": 5},
    {"lr": 0.0005, "batch_size": 64, "epochs": 5},
    {"lr": 0.001, "batch_size": 128, "epochs": 5},
    {"lr": 0.001, "batch_size": 64, "epochs": 10},
]

# === 5. Experiment Loop ===
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

with open(summary_file_path, "w") as summary_file:
    summary_file.write("=== COMBINED EXPERIMENT SUMMARY (Task C + D) ===\n")

    experiment_id = 1

    for arch in architectures:
        for hparam in hyperparams:
            print(f"\n=== Experiment {experiment_id} ===")
            print(f"Architecture: {arch} | Hyperparams: {hparam}")

            # Load data with specific batch size
            train_loader, test_loader, _ = load_emnist_data(
                batch_size_train=hparam["batch_size"], batch_size_test=1000
            )

            # Initialize model and optimizer
            model = DeepCNN(**arch).to(device)
            optimizer = optim.Adam(model.parameters(), lr=hparam["lr"])
            criterion = nn.CrossEntropyLoss()

            all_batch_losses = []
            start_time = time.time()

            # Training loop
            for epoch in range(1, hparam["epochs"] + 1):
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

            end_time = time.time()
            training_time = end_time - start_time
            final_accuracy = test_acc(model, test_loader)

            # Plot loss
            loss_plot_path = os.path.join(output_dir_base, f"loss_exp{experiment_id}.png")
            plt.figure(figsize=(10, 5))
            for epoch_idx, losses in enumerate(all_batch_losses):
                smoothed = smooth_losses(losses)
                plt.plot(smoothed, label=f"Epoch {epoch_idx + 1}")
            plt.title(f"Loss Curve – Experiment {experiment_id}")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.close()

            # Save model
            model_name = generate_model_name(
                architecture="deepcnn",
                num_filters=arch["channels_2"],
                fc_out=26,
                num_epochs=hparam["epochs"],
                optimizer_name="Adam",
                learning_rate=hparam["lr"],
                accuracy=final_accuracy,
                device_used=str(device)
            )
            model_path = save_model(model, model_name, output_dir=model_dir_base)

            # Save detailed individual summary
            print_and_save_summary(
                device=device,
                num_epochs=hparam["epochs"],
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer_name="Adam",
                learning_rate=hparam["lr"],
                training_time=training_time,
                loss_plot_path=loss_plot_path,
                sample_plot_path=os.path.join(output_dir_base, "sample_images.png"),
                final_accuracy=final_accuracy,
                model_path=model_path,
                output_dir=output_dir_base
            )

            # Write to combined summary file
            summary_file.write(f"\n--- EXPERIMENT {experiment_id} ---\n")
            summary_file.write(f"Channels: {arch['channels_1']} → {arch['channels_2']}, Kernel Size: {arch['kernel_size']}\n")
            summary_file.write(f"Learning Rate: {hparam['lr']}, Batch Size: {hparam['batch_size']}, Epochs: {hparam['epochs']}\n")
            summary_file.write(f"Final Accuracy: {final_accuracy:.2f}%\n")
            summary_file.write(f"Training Time: {training_time:.2f} sec\n")
            summary_file.write(f"Model Path: {model_path}\n")
            summary_file.write(f"Loss Plot: {loss_plot_path}\n")

            experiment_id += 1
