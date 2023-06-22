from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



model = pickle.load(open('ml/model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    np_features = [np.array(float_features)]
    prediction = model.predict(np_features)
    prediction = prediction[0]

    return render_template('index.html', context=f'Predicted Class: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)