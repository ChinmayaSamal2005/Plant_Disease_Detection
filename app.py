from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import joblib

# Initialize Flask app
app = Flask(__name__,template_folder='template')

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path = r"C:\Plant-Disease-Detection\Model\plant_disease_model.h5"
model = load_model(model_path)

# Load class labels
labels_path = r"C:\Plant-Disease-Detection\Model\labels.pkl"
all_labels = joblib.load(labels_path)

# Function to preprocess uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))  # must match training size
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    image_url = ""
    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Preprocess and predict
        img = preprocess_image(image_path)
        pred = model.predict(img)
        predicted_class = all_labels[np.argmax(pred)]
        prediction = f"🧠 Prediction: {predicted_class}"
        image_url = image_path

    return render_template("index.html", prediction=prediction, image=image_url)

if __name__ == "__main__":
    app.run(debug=True,port=5003)