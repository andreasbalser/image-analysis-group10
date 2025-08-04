Our initial model is very simple. Your task is to define and train a new, more complex model by adding more layers. Document your observations. How do changes in the architecture of the model effect the accuracy and training duration?

You should also experiment with changing the number of output channels (feature maps) and the size of the convolution kernel. Document your experiment results and briefly explain the meaning of feature maps.

Some of the parameters relevant to duration of the training process and the prediction accuracy of a network include learning rate, number of epochs, batch size. Test different variations of these parameters. What is the purpose of these parameters? How does varying the learning rate, number of epochs and batch size change the learning behavior?

---
# Task B – Architecture and Hyperparameter Experiments

After the initial baseline in Task A, we expanded our experiments by training a more complex CNN architecture and systematically varying model and training parameters.

---

## 🔧 Model Architecture: DeepCNN

The enhanced model (`DeepCNN`) added:
- A second **Conv2D → ReLU → MaxPool** block
- A **Dropout** layer to reduce overfitting
- A **128-unit fully connected (FC) layer** before classification

This made the model significantly deeper and more expressive.

---

## 🔍 Observations on Architecture Changes

| Variant | Channels (c1 → c2) | Kernel Size | Accuracy (%) | Time (s) |
|---------|--------------------|--------------|---------------|-----------|
| Exp 1–4 | 8 → 16             | 3×3          | 93.14 max     | 90–2133   |
| Exp 5–8 | 16 → 32            | 3×3          | 93.43 max     | 130–292   |
| Exp 9–12| 32 → 64            | 3×3          | **93.72**     | 236–496   |
| Exp 13–16| 16 → 32           | **5×5**      | **93.92**     | 223–493   |

### ✅ Key Takeaways:
- **Deeper networks** (more filters, larger kernels) generally performed better.
- Increasing the number of filters (output channels) improved feature extraction, especially when combined with longer training.
- **5×5 kernels** improved accuracy, likely by capturing more spatial context.
- Larger models increased training time.

---

## 📘 What Are Feature Maps?

Feature maps are the outputs of convolutional layers. Each map highlights different patterns (e.g., edges, corners, shapes). Increasing the number of output channels means the network can learn **more diverse and abstract features**, improving accuracy — especially in complex tasks like handwriting recognition.

---

## ⚙️ Hyperparameter Experiments

We tested different combinations of:

| Parameter       | Values tested                          |
|-----------------|-----------------------------------------|
| Learning rate   | `0.001`, `0.0005`                      |
| Batch size      | `64`, `128`                            |
| Epochs          | `5`, `10`                              |

### 💡 Observations:

- **Learning rate**:
  - `0.001` generally converged faster and reached higher accuracy.
  - `0.0005` was slower and sometimes underperformed due to smaller updates.

- **Batch size**:
  - `64` gave better performance — more frequent updates and generalization.
  - `128` was slightly faster but occasionally led to lower accuracy.

- **Epochs**:
  - Longer training (10 epochs) **almost always improved accuracy**.
  - Especially important for deeper or more complex architectures.

---

## 🏆 Best Result

- **Experiment 13** (16 → 32 channels, 5×5 kernel, lr=0.001, bs=64, 5 epochs)
- **Final Accuracy**: **93.92%**
- **Training Time**: 233.45 sec

This model was saved for use in Task C and Task D.

---
