from flask import Flask, render_template, request
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
    sqft = float(request.form['sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    
    # Prepare the input features for prediction
    loc_index = np.where(columns == location)[0][0]
    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    # Make prediction
    prediction = model.predict([x])[0]*100000

    # Render the predicted result page with the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


