Training Configuration and Results

Dataset: EMNIST (letters split)
Input size: 28×28 grayscale images
CNN architecture:
Conv2d(1 → 8 filters, kernel size 3, padding 1)
ReLU
MaxPool2d (2×2)
Flatten
Linear(8×14×14 → 26 output classes)
Loss Function: CrossEntropyLoss
Optimizer: Adam
Chosen for its adaptive learning rate and good default behavior in deep learning.
Learning Rate: 0.001
Batch Size: 64
Epochs: 5
Device used: {{ GPU if torch.cuda.is_available() else CPU }}
Final Test Accuracy: 88.63%
Total Training Time: 43.77 seconds 

