import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load CNN model
model = load_model('../model/cnn_model.h5')

IMG_SIZE = 64

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

# -------------------------------
# Routes
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None
    confidence = None 

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)

            prediction = model.predict(img)[0][0]
            confidence = round(float(prediction), 3)


            if prediction > 0.5:
                result = "Stego Image"
            else:
                result = "Clean Image"
            print("Prediction value:", prediction)
            
            image_path = filepath

    return render_template('index.html', result=result, image_path=image_path, confidence=confidence)
@app.route('/graph')
def graph():
    history = joblib.load('../model/history.pkl')

    accuracy = history['accuracy']
    epochs = list(range(1, len(accuracy)+1))

    return render_template(
        'graph.html',
        accuracy=accuracy,
        epochs=epochs
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
