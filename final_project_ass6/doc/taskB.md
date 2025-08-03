Our initial model is very simple. Your task is to define and train a new, more complex model by adding more layers. Document your observations. How do changes in the architecture of the model effect the accuracy and training duration?

You should also experiment with changing the number of output channels (feature maps) and the size of the convolution kernel. Document your experiment results and briefly explain the meaning of feature maps.

Some of the parameters relevant to duration of the training process and the prediction accuracy of a network include learning rate, number of epochs, batch size. Test different variations of these parameters. What is the purpose of these parameters? How does varying the learning rate, number of epochs and batch size change the learning behavior?

---
# Task B – Network Design and Parameter Experiments

After completing Task A with a **simple baseline CNN**, we extended the experiments to explore **deeper architectures, more feature maps, different kernel sizes, and hyperparameter variations**. The aim was to improve performance on the EMNIST **letters** dataset.

---

## 1. Adding Layers – More Complex Model

In Task A, the CNN was shallow, with a single convolutional layer.  
For Task B, we implemented a **deeper CNN** with this structure:

- **Conv2D (channels\_1 filters, kernel)** → ReLU → MaxPooling  
- **Conv2D (channels\_2 filters, kernel)** → ReLU → MaxPooling  
- Flatten  
- Dropout (0.3)  
- Fully Connected (128 units) → ReLU  
- Output Layer (26 classes)

This **additional convolutional layer** enables the model to extract **hierarchical features**, improving its ability to generalize.

---

### **Impact of a Deeper Network**

- **Accuracy:**  
  - Increased significantly from **87.24% (Task A)** to **93–94% (Task B)**.
- **Training Time:**  
  - **Task A:** 43.66 seconds (1 conv layer)  
  - **Task B:** From ~85 seconds (small networks) up to 1,496 seconds (large, deep networks).
  - Example: Experiment 16 (best model) required **442.42 seconds**, which is roughly **10 times longer than Task A**.

---

## 2. Feature Maps (Output Channels)

**Feature maps** represent different learned filters that extract distinct patterns (edges, shapes, textures) in the image.  
Increasing the number of channels means the network learns **more patterns**, but also requires **more computation**.

---

### **Effects of Changing Channels and Kernel Size**

- **Channels:**
  - Increased from **8→16** to **32→64**, resulting in higher accuracy.
- **Kernel size:**
  - **3×3 kernels**: Efficient for fine-grained local patterns.
  - **5×5 kernels**: Capture a larger context, sometimes improving accuracy (e.g., 93.97% with 5×5).

---

## 3. Hyperparameter Experiments

We systematically varied:

- **Learning Rate (LR)**: 0.001 vs 0.0005
- **Batch Size**: 64 vs 128
- **Epochs**: 5 vs 10

### **Purpose and Observations:**

- **Learning Rate (LR)**  
  - Controls step size during optimization.  
  - **0.001** → faster convergence, slightly better results.  
  - **0.0005** → safer convergence, but slower and less accurate.

- **Batch Size**  
  - Defines how many samples are processed before updating weights.  
  - **64:** More frequent updates, better generalization.  
  - **128:** Smoother gradients but slightly lower accuracy.

- **Epochs**  
  - **More epochs** improve accuracy at the cost of increased training time.

---

### **Impact on Training Time**

- **5 epochs:** 80–220 seconds depending on architecture.  
- **10 epochs:** Training time doubled or tripled, with best results:
  - Example: Experiment 16 → **442.42 seconds**.
- **Largest model:** 1,496 seconds (32→64 channels, 10 epochs).

---

## 4. Results Overview

### **Best Experiment (Exp. 16)**

- Channels: 16 → 32  
- Kernel Size: 5×5  
- Learning Rate: 0.001  
- Batch Size: 64  
- Epochs: 10  
- **Final Accuracy:** 93.97%  
- **Training Time:** 442.42 seconds  

---

## 5. Optimizer

**Adam optimizer** was chosen because it combines momentum with adaptive learning rates, leading to **fast and stable convergence without much tuning**.

---

## 6. Final Model

The **best-performing model** was automatically saved for later tasks:


This model will be used for:

- **Transfer learning (Task C)**
- **Classifying own test images (Task D)**

---

## 7. Key Findings

- **Deeper architectures improve accuracy** significantly.
- **More channels and larger kernels** improve feature extraction.
- **More epochs and smaller batch sizes** improve performance, but increase training time.
- Training time increased from **~44 s (Task A)** to **442 s (Task B best model)**.

---

## Conclusion

Experiments confirmed that deeper CNNs with carefully tuned hyperparameters outperform the baseline model. While accuracy improved from 87% to 94%, this came at the cost of longer training times. The final saved model balances performance and complexity and will serve as the foundation for the next tasks.

