from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
import joblib

app = Flask(__name__)


# Load the trained models
vgg_model = tf.keras.models.load_model("VGG_Plant_Diseases.h5")
inception_model = tf.keras.models.load_model("InceptionV3_Plant_Diseases_updated.h5")
densenet_model = tf.keras.models.load_model("DenseNet_Plant_Diseases.h5")
class_mapping ={
        0: 'Apple___Apple_scab', 
        1: 'Apple___Black_rot', 
        2: 'Apple___Cedar_apple_rust', 
        3: 'Apple___healthy', 4: 'Blueberry___healthy', 
        5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy', 
        7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_', 
        9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 
        11: 'Grape___Black_rot', 12: 'Grape___Esca_(Black_Measles)', 
        13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy', 
        15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 
        17: 'Peach___healthy', 18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 
        20: 'Potato___Early_blight', 21: 'Potato___Late_blight', 22: 'Potato___healthy', 
        23: 'Raspberry___healthy', 24: 'Soybean___healthy', 25: 'Squash___Powdery_mildew', 
        26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy', 28: 'Tomato___Bacterial_spot', 
        29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight', 31: 'Tomato___Leaf_Mold', 
        32: 'Tomato___Septoria_leaf_spot', 33: 'Tomato___Spider_mites Two-spotted_spider_mite', 
        34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
        36: 'Tomato___Tomato_mosaic_virus',
        37: 'Tomato___healthy'
        }

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize for model compatibility
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def ensemble_prediction(image_path):
    img = preprocess_image(image_path)

    # Get predictions from each model
    vgg_pred = vgg_model.predict(img)
    inception_pred = inception_model.predict(img)
    densenet_pred = densenet_model.predict(img)

    # Average the predictions (ensemble technique)
    final_pred = (vgg_pred + inception_pred + densenet_pred) / 3
    predicted_class = np.argmax(final_pred, axis = 1)
    predicted_class_index = predicted_class[0]  # Get the scalar value (first element)
    predicted_class_name = class_mapping.get(predicted_class_index, "Unknown Class")
    return predicted_class_name

@app.route('/')
def home():
    return render_template('sample1.html')

@app.route('/sample2')
def crop_recommendation():
    return render_template('sample2.html')

@app.route('/leaf_det')
def leaf_detection():
    return render_template('leaf_det.html')

@app.route('/predict_leaf', methods=['POST'])
def predict_leaf():
    if 'leaf_image' not in request.files:
        return "No file uploaded", 400

    file = request.files['leaf_image']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded image
    file_path = "static/uploads/" + file.filename
    file.save(file_path)

    # Get prediction
    prediction = ensemble_prediction(file_path)

    return render_template('leaf_det.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
