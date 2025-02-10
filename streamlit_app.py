import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("PhytovisionModel.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Phytovision v1")
st.write("Upload an image of a fruit or vegetable to classify it.")
st.write("Please note this is a very early prototype and the model might make some mistakes. I am working on improving accuracy now that the foundations are almost complete")
st.write("ResNet18 model created with the help of PyTorch and trained on a dataset from Kagglehub")
st.write("I am working on Phytovision v2 that uses a similar architecture to this with added improvements and will be trained on a plant disease dataset")
 

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
    
    st.write(f"### Prediction: {label}")
