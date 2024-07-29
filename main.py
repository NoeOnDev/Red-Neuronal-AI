from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)
model = load_model('picas_model.h5')

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
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    img_array = cv2.imread(file_path)
    if img_array is None:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': 'Invalid image'}), 400
    
    img_array = preprocess_image(img_array)
    prediction = model.predict(img_array)
    
    classes = ['As de picas', 'Dos de picas', 'Tres de picas', 'Cuatro de picas', 'Cinco de picas', 'Seis de picas']
    confidence_threshold = 0.5

    max_prediction = np.max(prediction)
    if max_prediction < confidence_threshold:
        predicted_class = "No hay carta reconocida"
    else:
        predicted_class = classes[np.argmax(prediction)]
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
