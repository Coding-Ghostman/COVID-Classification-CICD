import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define data directory
test_dir = 'Data'  # Assuming you have a 'test' folder with COVID and Normal subfolders

# Load the trained model using pickle
with open('covid_classification_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to process and predict the test images
def predict_test_images():
    X_test, y_test = [], []

    for label in os.listdir(test_dir):
        label_path = os.path.join(test_dir, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array)
            predicted_label = 1 if prediction > 0.5 else 0  # Assuming binary classification

            X_test.append(img_array)
            y_test.append(label)# Assuming subfolders are named as class labels
            print(f'''Original label: {label}\nPredicted label: {predicted_label}''')
            
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    return X_test, y_test

# Evaluate the model on the test set
def evaluate_model():
    X_test, y_test = predict_test_images()

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    classification_report_str = classification_report(y_test, y_pred_binary)
    confusion_mat = confusion_matrix(y_test, y_pred_binary)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report_str)
    print("Confusion Matrix:")
    print(confusion_mat)

if __name__ == "__main__":
    evaluate_model()
