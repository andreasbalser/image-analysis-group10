After loading the dataset, plot a random selection of image samples and the corresponding labels (similar to Figure 1.). Include this figure in your project documentation.

Briefly explain the meaning of the network layers and their parameters.

Record in your project documentation the achieved accuracy, all important training parameters such as number of epochs, batch size, total training time, as well as the plot of the loss during training. Explain which optimizer you have selected and why. Report whether the training was performed using the CPU or the GPU.

# Task A – Initial Setup and Baseline Training
The network architecture consisted of:

- conv2D (8 filters, 3×3 kernel, padding=1)Extracts local spatial features while preserving spatial dimensions 
- ReLU: Activation, introduces non-linearity  
- MaxPooling (2×2): Downgrading, reduces the feature map size from 28×28 to 14×14  
- Flattening: Converts the feature maps into a 1D vector  
- Fully Connected Layer (1568 → 26): Maps the extracted features to the alphabet classes
- CrossEntropyLoss was used as the loss function
- Adam optimizer was used for parameter updates. Adam was selected because it combines the benefits of momentum and adaptive learning rates (speeds up convergence)
- The training was performed on the available device (CPU)


## Training Configuration and Results

=== TRAINING SUMMARY ===
Device used: cpu
Number of epochs: 5
Training batch size: 64
Testing batch size: 1000
Optimizer: Adam
Learning rate: 0.001
Total training time: 61.30 seconds
Loss plot saved to: output_images/output_images_taskA/training_loss_plot_smoothed.png
Sample image plot saved to: output_images/output_images_taskA/sample_images.png
Final Test Accuracy: 88.16%
Model saved to: saved_models/saved_models_taskA/cnn1_8f_fc26_ep5_adam_lr0.001_acc88_cpu.pth

## Interpretation

- The loss decreased steadily across epochs, with the largest improvement during the first epoch, then gradual convergence
Epoch 1 — Avg. Training Loss: 0.8499
Test Accuracy: 83.85%
Epoch 2 — Avg. Training Loss: 0.4757
Test Accuracy: 86.93%
Epoch 3 — Avg. Training Loss: 0.4127
Test Accuracy: 87.40%
Epoch 4 — Avg. Training Loss: 0.3847
Test Accuracy: 87.97%
Epoch 5 — Avg. Training Loss: 0.3678
Test Accuracy: 88.16%

- A final test accuracy of 88.16% was achieved with a total training time of 61.30 seconds
- Given that the model was trained on CPU and with a shallow architecture, these results are promising

