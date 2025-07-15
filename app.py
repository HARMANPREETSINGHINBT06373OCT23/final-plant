import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

# ====== Model & Class Setup ======
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

classes = [
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_Spot", "Tomato_Yellow_Leaf_Curl", "Tomato_Healthy",
    "Potato_Early_Blight", "Potato_Late_Blight", "Potato_Healthy",
    "Corn_Common_Rust"
]

disease_info = {
    "Tomato_Bacterial_Spot": {"definition": "Bacterial spots on leaves and fruits.", "color": "red", "health_status": "Unhealthy"},
    "Tomato_Early_Blight": {"definition": "Fungal dark spots on older leaves.", "color": "brown", "health_status": "Unhealthy"},
    "Tomato_Leaf_Mold": {"definition": "Yellow spots with mold under leaves.", "color": "orange", "health_status": "Unhealthy"},
    "Tomato_Septoria_Spot": {"definition": "Gray-centered circular spots.", "color": "gray", "health_status": "Unhealthy"},
    "Tomato_Yellow_Leaf_Curl": {"definition": "Curling and yellowing of leaves.", "color": "yellow", "health_status": "Unhealthy"},
    "Tomato_Healthy": {"definition": "Healthy tomato leaf.", "color": "green", "health_status": "Healthy"},
    "Potato_Early_Blight": {"definition": "Brown spots with rings on leaves.", "color": "brown", "health_status": "Unhealthy"},
    "Potato_Late_Blight": {"definition": "Rapid lesions on leaves and stems.", "color": "darkred", "health_status": "Unhealthy"},
    "Potato_Healthy": {"definition": "Healthy potato foliage.", "color": "green", "health_status": "Healthy"},
    "Corn_Common_Rust": {"definition": "Red-brown pustules on corn leaves.", "color": "red", "health_status": "Unhealthy"}
}

# ====== Model Load (pretend-trained model) ======
device = torch.device("cpu")
model = TinyCNN(num_classes=len(classes)).to(device)

# Dummy training - optional: load a real pre-trained model
model.eval()

# ====== Image Preprocessing ======
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ====== Streamlit UI ======
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detector")
st.markdown("Upload a leaf image to get disease predictions.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]
        info = disease_info.get(predicted_class, {
            'definition': 'Unknown disease.',
            'color': 'gray',
            'health_status': 'Unknown'
        })

    st.subheader("🩺 Prediction Result:")
    st.markdown(f"**Disease:** `{predicted_class}`")
    st.markdown(f"**Status:** `{info['health_status']}`")
    st.markdown(f"**Details:** {info['definition']}")
