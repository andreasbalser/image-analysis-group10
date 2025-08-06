import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pretrained models with classes
LETTER_MODEL_PATH = "saved_models/saved_models_taskB/deepcnn_c116_c232_k5_lr0.001_bs64_ep5_acc94_cpu.pth"
DIGIT_MODEL_PATH = "saved_models/saved_models_taskC/deepcnn_transfer_c116_c232_k5_fc10_ep5_adam_lr0.001_bs64_acc99_cpu.pth"

LETTER_CLASSES = [chr(i) for i in range(65, 91)] 
DIGIT_CLASSES = [str(i) for i in range(10)]     

#preprocessing to maybe enhance results a little bit (spoiler: didnt really enhance it)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class DeepCNN(nn.Module):
    def __init__(self, channels_1=16, channels_2=32, kernel_size=5, num_classes=26):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, channels_1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(channels_1, channels_2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels_2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def load_model(model_path, num_classes):
    model = DeepCNN(channels_1=16, channels_2=32, kernel_size=5, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


#prediction with other image enhancements
def predict_image(image_path, model, class_labels):
    image = Image.open(image_path).convert("L") 
    image = ImageOps.autocontrast(image)        
    image = image.resize((28, 28))

    image = transform(image)                    
    image = image.unsqueeze(0).to(DEVICE) 
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_labels[predicted.item()]

def predict_folder(folder_path, model, class_labels):
    predictions = {}
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            pred = predict_image(img_path, model, class_labels)
            predictions[filename] = pred
    return predictions

if __name__ == "__main__":
    # Predict letters
    print("--- Letter predictions ---")
    letter_model = load_model(LETTER_MODEL_PATH, num_classes=26)
    letter_folder = "own_data/letters"
    letter_preds = predict_folder(letter_folder, letter_model, LETTER_CLASSES)
    for fname, pred in letter_preds.items():
        print(f"{fname}: {pred}")

    # Predict digits
    print("\n--- Digit predictions ---")
    digit_model = load_model(DIGIT_MODEL_PATH, num_classes=10)
    digit_folder = "own_data/numbers"
    digit_preds = predict_folder(digit_folder, digit_model, DIGIT_CLASSES)
    for fname, pred in digit_preds.items():
        print(f"{fname}: {pred}")
