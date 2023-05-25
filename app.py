# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
import joblib


# Load the Random Forest CLassifier model
filename = "model.pkl"
model = joblib.load(filename)

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        texture_mean= float(request.form['texture_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        symmetry_mean = float(request.form['fractal_dimension_mean'])
        texture_se = float(request.form['texture_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        concavity_se = float(request.form['concavity_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        smoothness_worst = float(request.form['smoothness_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        
        data = np.array([[texture_mean,area_mean,smoothness_mean,concavity_mean,symmetry_mean,texture_se,area_se,smoothness_se,concavity_se,symmetry_se,fractal_dimension_se,smoothness_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

