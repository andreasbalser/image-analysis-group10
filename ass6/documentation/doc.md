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
