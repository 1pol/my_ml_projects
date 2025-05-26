from flask import Flask, render_template, request
from houseprice import price_predict
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
model = pickle.load(open('model.pkl', 'rb'))

# Load the columns used for training the model
columns = pickle.load(open('columns.pkl', 'rb'))

df = pd.read_csv(r"C:\Users\KIIT0001\Desktop\myflaskproject\Bengaluru_House_Data.csv")
locations = df['location'].unique().tolist()

@app.route('/')
def home():
    with open('columns.pkl', 'rb') as f:
        data_columns = pickle.load(f)
    locations = data_columns[3:]  # Assuming first three columns are sqft, bath, bhk
    return render_template('index.html', locations=locations)
def index():
    return render_template('index.html',locations=locations)

def price_predict(location, sqft, bath, bhk):
    # Load model and columns
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('columns.pkl', 'rb') as f:
        data_columns = pickle.load(f)

    # Prepare input vector
    try:
        loc_index = data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    try:
        sqft = float(request.form['sqft'])
        location = request.form['location']
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])

        if sqft < 300 or bath <= 0 or bhk <= 0 or bhk > 20:
            return render_template('result.html', prediction="Invalid input. Please enter realistic values.")
    # Make prediction
        prediction = price_predict(location ,sqft , BHK)
        if prediction < 0:
            return render_template('result.html', prediction="Prediction resulted in a negative value. Please check your inputs.")
        return render_template('result.html', prediction=f"₹ {prediction:,.2f}")
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)


