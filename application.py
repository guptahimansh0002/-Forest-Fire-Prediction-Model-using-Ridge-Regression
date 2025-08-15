import pickle
from flask import Flask, request,jsonify,render_template

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
ridge_path  = os.path.join(BASE_DIR, "model", "ridge.pkl")

# standard_scaler = pickle.load(open(scaler_path, 'rb'))
# ridge_model     = pickle.load(open(ridge_path, 'rb'))

# import pickle
# import os

# cwd=os.getcwd()
# scaler_path=os.path.join(cwd, "model","scaler.pkl")
# ridge_path=os.path.join(cwd, "model","ridge.pkl")

import joblib
standard_scaler = joblib.load(scaler_path)
ridge_model = joblib.load(ridge_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get("Temperature"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template("home.html",results=result[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)






# file_path1=r"D:\workspace\model\scaler.pkl"

# with open(file_path1,'rb') as s:
#     scaler = pickle.load(s)

# file_path2=r"D:\workspace\model\ridge.pkl"

# with open(file_path2,'rb') as r:
#     scaler = pickle.load(r)






# import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
# ridge_path  = os.path.join(BASE_DIR, "model", "ridge.pkl")

# standard_scaler = pickle.load(open(scaler_path, 'rb'))
# ridge_model     = pickle.load(open(ridge_path, 'rb'))

# import joblib
# standard_scaler = joblib.load(scaler_path)
# ridge_model = joblib.load(ridge_path)

# import os
# import joblib

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
# ridge_path  = os.path.join(BASE_DIR, "model", "ridge.pkl")