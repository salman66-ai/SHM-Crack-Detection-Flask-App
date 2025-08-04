from flask import Flask, request, render_template, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = 'crack_detection_pretrained.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Helper function to preprocess the image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Route for the home page
@app.route('/')
def upload_file():
    return render_template('upload.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Save the uploaded file in the 'uploads' directory
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Predict the result
        img_array = prepare_image(file_path)
        prediction = model.predict(img_array)[0][0]
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        result = "Not Cracked" if prediction > 0.5 else "Cracked"

        return render_template('result.html', result=result, confidence=confidence, img_path=file.filename)

# Serve uploaded images from the 'uploads' folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    # Create 'uploads' directory if not exists
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
