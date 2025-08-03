After loading the dataset, plot a random selection of image samples and the corresponding labels (similar to Figure 1.). Include this figure in your project documentation.

Briefly explain the meaning of the network layers and their parameters.

Record in your project documentation the achieved accuracy, all important training parameters such as number of epochs, batch size, total training time, as well as the plot of the loss during training. Explain which optimizer you have selected and why. Report whether the training was performed using the CPU or the GPU.

---

# Task A – Initial Setup and Baseline Training

For the initial setup, the EMNIST **letters** dataset (28×28 grayscale images) was used to train a simple CNN for handwritten letter classification. The network architecture consisted of:

- **Conv2D (8 filters, 3×3 kernel)**: Extracts local spatial features  
- **ReLU**: Introduces non-linearity  
- **MaxPooling (2×2)**: Reduces feature map size from 28×28 to 14×14  
- **Flattening**: Converts feature maps into a 1D vector  
- **Fully Connected Layer (1568 → 26)**: Maps extracted features to the 26 alphabet classes  

**CrossEntropyLoss** was used as the loss function and **Adam optimizer** for parameter updates. Adam was selected because it combines the benefits of momentum and adaptive learning rates, which typically speeds up convergence. The training was done on the CPU.

---

## Training Configuration and Results

Device used: cpu
Number of epochs: 5
Training batch size: 64
Testing batch size: 1000
Optimizer: Adam
Learning rate: 0.001
Total training time: 43.66 seconds
Loss plot saved to: final_project_ass6/output_images/output_images_taskA/training_loss_plot_smoothed.png
Sample image plot saved to: final_project_ass6/output_images/output_images_taskA/sample_images.png
Final Test Accuracy: 87.24%
Model saved to: final_project_ass6/saved_models/saved_models_taskA/cnn1_8f_fc26_ep5_adam_lr0.001_acc87_cpu.pth

---

## Interpretation

- The **loss decreased steadily across epochs**, with the largest improvement during the first epoch, then gradual convergence.
- Epoch 1 — Avg. Training Loss: 0.9847; Test Accuracy: 78.55%
- Epoch 2 — Avg. Training Loss: 0.5917; Test Accuracy: 84.83%
- Epoch 3 — Avg. Training Loss: 0.4711; Test Accuracy: 86.16%
- Epoch 4 — Avg. Training Loss: 0.4224; Test Accuracy: 87.13%
- Epoch 5 — Avg. Training Loss: 0.3947; Test Accuracy: 87.24%

- A **final test accuracy of 87.24%** was achieved using a simple CNN architecture.
- Given that the model was trained on CPU and with a shallow architecture, these results are promising, but there is room for improvement.  

