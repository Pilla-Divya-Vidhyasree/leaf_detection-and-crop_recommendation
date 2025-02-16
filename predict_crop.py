from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML model and label encoder
model = joblib.load("gaussiannb_crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Home page
@app.route('/')
def home():
    return render_template('sample1.html')

# Crop recommendation page
@app.route('/sample2')
def crop_recommendation():
    return render_template('sample2.html')

#Leaf Disease detectioon
@app.route('/leaf_det')
def leaf_detection():
    return render_template('leaf_det.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.form)  # Debugging line
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        return render_template('sample2.html', prediction=crop_name)
    except Exception as e:
        return f"Error: {e}", 400  # Show error message properly

if __name__ == '__main__':
    app.run(debug=True)

