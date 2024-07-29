from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)
model = load_model('as_de_picas_model.h5')

def preprocess_image(img_array):
    img_array = cv2.resize(img_array, (150, 150))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_path = './' + file.filename
    file.save(file_path)
    
    img_array = cv2.imread(file_path)
    img_array = preprocess_image(img_array)
    prediction = model.predict(img_array)
    
    predicted_class = "As de picas" if prediction[0] > 0.5 else "No As de picas"
    
    os.remove(file_path)
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
