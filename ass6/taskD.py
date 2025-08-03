import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from PIL import ImageOps

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to pretrained models
LETTER_MODEL_PATH = "ass6/saved_models/final_models/letter_model.pth"
DIGIT_MODEL_PATH = "ass6/saved_models/final_models/transfer_digit_model.pth"

# Model settings
LETTER_CLASSES = [chr(i) for i in range(65, 91)]  # A-Z
DIGIT_CLASSES = [str(i) for i in range(10)]       # 0-9

# Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# Model architecture
# (same as in Task B/C)
# -----------------------------
class DeepCNN(nn.Module):
    def __init__(self, channels_1=32, channels_2=64, kernel_size=3, num_classes=26):
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
# Load model
# -----------------------------
def load_model(model_path, num_classes):
    model = DeepCNN(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------------
# Inference function
# -----------------------------
def predict_image(image_path, model, class_labels):
    # Open as grayscale PIL image
    image = Image.open(image_path).convert("L")

    # Enhance contrast while still a PIL image
    image = ImageOps.autocontrast(image)

    # Resize (if not already 28x28)
    image = image.resize((28, 28))

    # Convert to tensor
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
    # Load both models
    letter_model = load_model(LETTER_MODEL_PATH, num_classes=26)
    digit_model = load_model(DIGIT_MODEL_PATH, num_classes=10)

    # Predict on own test images
    letter_folder = "ass6/own_data/letters"
    digit_folder = "ass6/own_data/numbers"

    letter_preds = predict_folder(letter_folder, letter_model, LETTER_CLASSES)
    digit_preds = predict_folder(digit_folder, digit_model, DIGIT_CLASSES)

    print("Letter predictions:")
    for fname, pred in letter_preds.items():
        print(f"{fname}: {pred}")

    print("\nDigit predictions:")
    for fname, pred in digit_preds.items():
        print(f"{fname}: {pred}")
