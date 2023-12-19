import os 
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torchvision import models

app = Flask(__name__)

# Configure file upload
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "upload"
configure_uploads(app, photos)

# Load the trained DenseNet model
model = models.densenet121(pretrained=False)
num_classes = 2  # Modify this based on the number of classes in your model
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load("Model/covid_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# Function to process and predict the uploaded image
def predict_image(file_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img = Image.open(file_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)

    probabilities = torch.softmax(output, dim=1)
    prediction = probabilities[:, 1].item()  # Probability of being in class 1 (COVID)
    
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "photo" in request.files:
        photo = request.files["photo"]
        if photo:
            # filename = secure_filename(photo.filename)
            filename = "photo.png"
            file_path = os.path.join("static/upload", filename)
            photo.save(file_path)

            # Perform prediction
            prediction = predict_image(file_path)

            # Display the result
            result = "COVID-19 Positive" if prediction < 0.7 else "COVID-19 Negative"

            return render_template("index.html", filename=filename, result=result)

    return render_template("index.html", filename=None, result=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)  # Local host
    # app.run(host='0.0.0.0', port=8080)  # For AWS
