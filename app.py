import pickle
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

import risk_recommendation

RANDOM_FOREST = "models/random_forest.pkl"
SCALER_FILE = 'models/scaler.sav'
JSON_LABEL = "models/lable_data.txt"
DATA_FILE = "test1.txt"
LIST_BIOMARKERS = ["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]
categorical_features = ["gender","ever_married","work_type","Residence_type","smoking_status"]
lables = {}

with open(JSON_LABEL, "r") as json_file:
    lables = json.load(json_file)

# load model from .pkl file
model = pickle.load(open(RANDOM_FOREST, 'rb'))

def preprocess_data(pred_arr):
    X = pd.DataFrame(pred_arr, index=[0])
    # load scaler object
    scaler = pickle.load(open(SCALER_FILE, 'rb'))
    # load model from .pkl file
    for item in categorical_features:
        idx = lables[item].index(X.loc[0, item])
        X.loc[0, item] = idx
    # scale the data
    X_scaler = scaler.transform(X)
    return X_scaler

def make_prediction(data):
    # run prediction
    pred_prob = model.predict_proba(data)
    # fetch predictions
    non_stroke, stroke = pred_prob[0]
    result = round((stroke * 100), 2)
    feature_impartances = model.feature_importances_
    return str(result), feature_impartances

## Flask api
app = Flask(__name__)
CORS(app)

@app.route('/predict/stroke', methods=['POST'])
def infer_image():
    data = request.json
    # Prepare the dataframe
    df = preprocess_data(data)

    predictions, feature_importances = make_prediction(df)
    # Return on a JSON format
    return jsonify(
        prediction= predictions,
        feature_importances=feature_importances.tolist(),
        risks=risk_recommendation.stroke['risks'],
        recommendations=risk_recommendation.stroke['recommendations']
    )
