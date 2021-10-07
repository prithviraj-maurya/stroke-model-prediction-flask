import json
import pickle
import pandas as pd

## Constants
SCALER_FILE = 'models/scaler.sav'
DATA_FILE = "test1.txt"
LIST_BIOMARKERS = ["gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status"]
categorical_features = ["gender","ever_married","work_type","Residence_type","smoking_status"]
lables = {}
RANDOM_FOREST = "models/random_forest.pkl"
JSON_LABEL = "models/lable_data.txt"

with open(JSON_LABEL, "r") as json_file:
    lables = json.load(json_file)

# load model from .pkl file
rf_stroke_model = pickle.load(open(RANDOM_FOREST, 'rb'))

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

def make_stroke_prediction(data):
    # Prepare the dataframe
    df = preprocess_data(data)
    # run prediction
    pred_prob = rf_stroke_model.predict_proba(df)
    # fetch predictions
    non_stroke, stroke = pred_prob[0]
    result = round((stroke * 100), 2)
    feature_impartances = rf_stroke_model.feature_importances_
    return str(result), feature_impartances
