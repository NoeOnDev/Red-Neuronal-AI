import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

model = load_model('pastillas_model.h5')

def preprocess_image(img_path):
    img_array = cv2.imread(img_path)
    img_array = cv2.resize(img_array, (150, 150))
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

test_dir = 'images_test'
classes = ['01', '02', '03', '04']
class_indices = {cls: idx for idx, cls in enumerate(classes)}

true_labels = []
predictions = []

for class_name in classes:
    class_dir = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        if not img_path.endswith('.jpg'):
            continue
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        true_labels.append(class_indices[class_name])
        predictions.append(predicted_label)

cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
