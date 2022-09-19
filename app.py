import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,flash,session,escape
import numpy as np
import pandas as pd

app=Flask(__name__)

### Loaded the model 
model=pickle.load(open('regressionmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    data=request.json['data']