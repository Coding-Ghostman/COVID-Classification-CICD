import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import joblib
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

# Set the upload folder
upload_folder = "static/upload"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
app.config["UPLOAD_FOLDER"] = upload_folder

# Load the pickled model
with open("Model/covid_classifier.pkl", "rb") as model_file:
    model = joblib.load(model_file)

# Function to process and predict the uploaded image
def predict_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img = np.array(img.resize((256, 256))) / 255.0  # Normalize pixel values
    img = img.reshape(1, -1)

    prediction = model.predict_proba(img)[:, 1].item()  # Probability of being in class 1 (COVID)

    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "photo" in request.files:
        photo = request.files["photo"]
        if photo:
            filename = secure_filename(photo.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
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
