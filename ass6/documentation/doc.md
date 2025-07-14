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

