## Transfer Learning Results – Documentation

### 🕒 Training Duration
The transfer learning process took **151.79 seconds** to complete training over 5 epochs.

### 🎯 Test Accuracy
The final model achieved a **test accuracy of 98.75%** on the evaluation dataset.

### 📉 Training Loss per Epoch
- **Epoch 1:** Loss = 0.1051  
- **Epoch 2:** Loss = 0.0618  
- **Epoch 3:** Loss = 0.0616  
- **Epoch 4:** Loss = 0.0623  
- **Epoch 5:** Loss = 0.0615

### 💾 Saved Model Path
`saved_models/saved_models_taskC/deepcnn_transfer_c116_c232_k5_fc10_ep5_adam_lr0.001_bs64_acc99_cpu.pth`

### 📊 Performance Analysis
Compared to full training of all layers from scratch, the transfer learning approach significantly reduced training time while maintaining high accuracy.  
This efficiency is due to leveraging pre-trained convolutional layers and only fine-tuning the final classifier layers, which drastically reduces the number of trainable parameters and computational cost.
