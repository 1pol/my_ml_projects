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
def index():
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    try:
        sqft = float(request.form['sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])

    # Make prediction
        output = price_predict(location ,sqft , BHK)
    except Exception as e:
        output = f"Invalid input: {str(e)}"

    # Render the predicted result page with the prediction
    return render_template('result.html', prediction = output)

if __name__ == '__main__':
    app.run(debug=True)


