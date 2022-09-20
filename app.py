import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,flash,session,escape
import numpy as np
import pandas as pd

app=Flask(__name__)

### Loaded the model 
model=pickle.load(open('regressionmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])
if __name__=="__main__":
    app.run(debug=True)