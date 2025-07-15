# EMNIST Model Architecture Comparison – Task A vs Task B

## Overview

This document compares the training results of a simple CNN (Task A) and a deeper CNN (Task B) trained on the EMNIST letters dataset.

| Aspect                   | Task A (Simple CNN)                     | Task B (Deep CNN)                          |
|--------------------------|------------------------------------------|---------------------------------------------|
| **Model**                | 1 Conv Layer, ReLU, MaxPool, Linear      | 2 Conv Layers, ReLU, MaxPool ×2, Dropout, Linear ×2 |
| **Training Time**        | 43.93 seconds                            | 128.80 seconds                              |
| **Final Test Accuracy**  | 87.24%                                   | 93.26%                                      |
| **Device**               | CPU                                      | CPU                                          |

---

## Observations

### Accuracy
- The deeper CNN (Task B) achieved **93.26% accuracy**, compared to **87.24%** for the simpler CNN in Task A.
- The added layers allowed the model to learn more complex visual patterns from the EMNIST letter data.

### Training Time
- Training time increased from ~44 seconds to ~129 seconds, due to more convolutional and fully connected layers.
- The increase is expected and more pronounced on CPU.

### Overfitting
- Despite increased model complexity, no strong overfitting was observed.
- This is likely due to:
  - Inclusion of a **dropout layer**
  - A balanced and sufficiently large dataset (EMNIST)
  - A reasonable model depth

---

## Conclusion

- **Deeper architectures improve performance** significantly on the EMNIST classification task.
- The trade-off is **longer training time**, but accuracy gains justify the complexity.
- The deeper model from Task B serves as a strong baseline for future experiments (e.g., learning rate tuning, batch normalization, or GPU acceleration).


# CNN Architecture Experiments – Task B.C

## What Are Feature Maps?

In Convolutional Neural Networks (CNNs), **feature maps** (also called channels or filters) are the outputs of convolution layers. Each filter detects specific features like edges, textures, or shapes within the input.

- **More feature maps** allow the model to detect a broader range of patterns.
- However, increasing the number of filters also leads to **higher computational cost and memory usage**.

---

## Experiment Setup

We ran 4 model configurations, each trained on the EMNIST **letters** dataset with the following settings:

- **Epochs**: 5  
- **Optimizer**: Adam  
- **Learning Rate**: 0.001  
- **Batch Size**: 64 (training), 1000 (testing)  
- **Loss Function**: CrossEntropyLoss  
- **Device**: CPU  

All models used the same training code and architecture structure, with variations in filter size and kernel size.

---

## Results Summary

| Config | Channels (Conv1 → Conv2) | Kernel Size | Final Accuracy | Training Time |
|--------|---------------------------|-------------|----------------|----------------|
| 1      | 8 → 16                    | 3           | 92.79%         | 89.32 sec      |
| 2      | 16 → 32                   | 3           | 93.10%         | 133.72 sec     |
| 3      | 32 → 64                   | 3           | 93.33%         | 224.98 sec     |
| 4      | 16 → 32                   | 5           | 93.24%         | 225.02 sec     |

---

## Observations

- Increasing the number of feature maps improved accuracy across the first three configurations.
- **Diminishing returns**: The accuracy gain from 32→64 filters was minimal, while training time nearly doubled.
- Larger kernel size (**Config 4**) slightly improved accuracy but incurred **significantly more computation**.
- **Best trade-off**: Config 2 (16 → 32 filters, kernel size 3) offers a strong balance between performance and speed.

---

## Conclusion

- **Feature maps are essential** for detecting patterns — more maps = richer representations.
- **Larger filters** can help but should be used sparingly due to their cost.
- **Careful tuning** of architecture is necessary to balance model complexity and efficiency.

Each model, loss plot, and log is saved for further comparison.

# Combined Experiment Summary (Task C + D)

## Purpose of Parameters

### Learning Rate
- **Purpose**: Controls the size of weight updates during training.
- **Effect**:
  - Too high → risk of overshooting and divergence.
  - Too low → slow convergence, risk of getting stuck in local minima.
- **Example**: 
  - Exp. 1 (LR=0.001) vs. Exp. 2 (LR=0.0005) → Accuracy difference: 92.43% vs. 92.48%.
  - Exp. 12 (LR=0.001) outperforms Exp. 10 (LR=0.0005) → 94.06% vs. 93.01%.

### Epochs
- **Purpose**: Defines how many times the model processes the full dataset.
- **Effect**:
  - More epochs → allows more learning, may improve accuracy.
  - Too many epochs → risk of overfitting, longer training time.
- **Example**:
  - Exp. 4 (5 → 10 epochs) → Accuracy improves from 92.43% to 93.48%.
  - Exp. 8 and 12 (10 epochs) achieve top accuracies (93.72%, 94.06%).

### Batch Size
- **Purpose**: Number of samples processed before updating weights.
- **Effect**:
  - Smaller batch size → more frequent updates, better generalization, slower training.
  - Larger batch size → faster per epoch, less generalization, smoother gradients.
- **Example**:
  - Exp. 5 vs. 7: Batch size 64 (93.37%) outperforms batch size 128 (92.25%).
  - Exp. 9 vs. 11: Batch size 64 (93.51%) slightly better than 128 (93.36%).

---

## Best Performing Model (for Task C and Task D)

| Experiment | Channels     | Kernel Size | LR     | Batch Size | Epochs | Accuracy | Training Time |
|------------|--------------|-------------|--------|-------------|--------|----------|----------------|
| **12**     | 32 → 64      | 3           | 0.001  | 64          | 10     | **94.06%** | 440.32 sec     |

- **Model Path**: `ass6/saved_models/saved_models_taskB/deepcnn_64f_fc26_ep10_adam_lr0.001_acc94_cpu.pth`
- **Loss Plot**: `ass6/output_images/output_images_taskB/loss_exp12.png`

> **Note**: Include only this model in your submission folder.  
> This model will be used as the basis for **Task C** and **Task D**.
