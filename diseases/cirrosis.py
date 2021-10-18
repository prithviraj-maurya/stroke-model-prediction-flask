#
# Model itself for prediction (presence of cirrosis of the liver); model was buit on data of 5214 people.
# its input:
# Age (years) Gender (male = 1, female = 0)
# Weight (cm) Height (cm)
# Body Mass Index Waist (cm)
# Maximum Blood Pressure (systolic blood pressure) (mm/Hg) Minimum Blood Pressure (diastolic blood pressure) (mm/Hg)
# Total Cholesterol (mg/dL) PVD (peripheral vascular disease) (yes = 1, no = 0)
# Alcohol Consumption (yes = 1, no = 0) HyperTension (yes = 1, no = 0)
# Diabetes (yes = 1, no = 0) Hepatitis (yes = 1, no = 0)
# Family Hepatitis (yes = 1, no = 0) Chronic Fatigue (yes = 1, no = 0)
# * All gathered features (except PVD, Alcohol Consumption, HyperTension, Diabetes, Hepatitis, Family Hepatitis, Chronic Fatigue - it's binary feature) must be transformed by scaler before input
#

# Scores of the model:
# Model gives an accuracy on cross validation of 79 % and 81 % on test dataset



## Imports
from pickle import load
import numpy as np

with open("models/Cirrosis.pkl", "rb") as f:
    description, model, scaler = load(f)

# sample=
# [
# Age,
# Gender,
# Weight,
# Height,
# Body Mass Index,
# Waist,
# Maximum Blood Pressure,
# Minimum Blood Pressure,
# Total Cholesterol,
# PVD,
# Alcohol Consumption,
# HyperTension,
# Diabetes,
# Hepatitis,
# Family Hepatitis,
# Chronic Fatigue
# ]

def preprocess_data(pred_dict):
    sample = np.array(list(pred_dict.values()))
    sample = sample.reshape(1, -1)
    standard_X_without_cat_feat = scaler.transform(sample[:, 0:8])
    standard_X = np.concatenate((standard_X_without_cat_feat, sample[:, 8:]), axis=1)
    return standard_X

def make_cirrosis_prediction(data):
    # run prediction
    df = preprocess_data(data)
    pred_prob = model.predict_proba(df)
    # fetch predictions
    non_stroke, stroke = pred_prob[0]
    result = round((stroke * 100), 2)
    feature_impartances = []
    return str(result), feature_impartances