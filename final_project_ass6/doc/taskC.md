Record the duration of the transfer learning training and the average accuracy for your test data set. Did the training take significantly less time compared to the complete training all layers?


# Task C – Transfer Learning

For this task, transfer learning was applied to adapt the best-performing **letter classifier** (from Task B) for **digit classification (0–9)** using the **EMNIST digits** dataset.

---

## Procedure

1. **Dataset**  
   - Used the **EMNIST “digits” split** (10 classes, grayscale 28×28).

2. **Base model**  
   - Loaded the **best performing model** from Task B:
     ```
     deepcnn_32f_fc26_ep10_adam_lr0.001_acc94_cpu.pth
     ```

3. **Architecture changes**  
   - Replaced the **final fully connected layer** to output 10 classes instead of 26.

4. **Freezing layers**  
   - **All convolutional and intermediate layers were frozen**; only the final classification layer was trained.

5. **Training setup**  
   - Optimizer: **Adam** (learning rate = 0.001)  
   - Loss: CrossEntropyLoss  
   - Epochs: 5  
   - Batch size: 64  
   - Device: CPU  

---

## Results

### Training Progress

Epoch 1/5, Loss: 0.1359
Epoch 2/5, Loss: 0.0833
Epoch 3/5, Loss: 0.0829
Epoch 4/5, Loss: 0.0823
Epoch 5/5, Loss: 0.0835


### Performance Metrics

- **Training Time:** 146.68 seconds  
- **Final Test Accuracy:** 98.34 %

---

## Interpretation

- **Speed:** Training with transfer learning was **much faster** compared to training a model from scratch (Task B full model training took between 442 s and 1496 s).  
- **Accuracy:** The model reached **98.34 %**, which is **higher than the original letter classifier** and shows that reusing pretrained features is effective.
- **Conclusion:** Transfer learning significantly reduces training time while achieving excellent accuracy.
