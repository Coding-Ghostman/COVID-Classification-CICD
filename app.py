import os
from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure file upload
photos = UploadSet("photos", IMAGES)
app.config["UPLOADED_PHOTOS_DEST"] = "upload"
configure_uploads(app, photos)

# Load the trained model using pickle
with open('covid_classification_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to process and predict the uploaded image
def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    return prediction[0][0]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "photo" in request.files:
        photo = request.files["photo"]
        if photo:
            filename = secure_filename(photo.filename)
            file_path = os.path.join("upload", filename)
            photo.save(file_path)

            # Perform prediction
            prediction = predict_image(file_path)

            # Display the result
            result = "COVID-19 Positive" if prediction > 0.5 else "COVID-19 Negative"

            return render_template("index.html", filename=filename, result=result)

    return render_template("index.html", filename=None, result=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
