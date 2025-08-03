import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to pretrained letter model (best from Task B, Exp. 16)
LETTER_MODEL_PATH = "final_project_ass6/saved_models/saved_models_taskB/deepcnn_32f_fc26_ep10_adam_lr0.001_acc94_cpu.pth"

# Class labels for letters A–Z
LETTER_CLASSES = [chr(i) for i in range(65, 91)]  # A-Z

# Preprocessing: normalization as used during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# Model architecture (matches best Task B experiment)
# -----------------------------
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

# -----------------------------
# Load letter model
# -----------------------------
def load_letter_model(model_path):
    model = DeepCNN(channels_1=16, channels_2=32, kernel_size=5, num_classes=26)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------------
# Prediction function
# -----------------------------
def predict_image(image_path, model, class_labels):
    # Open and preprocess
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageOps.invert(image)  # Invert so text is white on black (like EMNIST)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    image = image.resize((28, 28))
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
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

# -----------------------------
# Run predictions
# -----------------------------
if __name__ == "__main__":
    # Load only the letter model
    letter_model = load_letter_model(LETTER_MODEL_PATH)

    # Folder with handwritten test images (letters only)
    letter_folder = "final_project_ass6/own_data/letters"

    # Predict and print results
    letter_preds = predict_folder(letter_folder, letter_model, LETTER_CLASSES)

    print("Letter predictions:")
    for fname, pred in letter_preds.items():
        print(f"{fname}: {pred}")
