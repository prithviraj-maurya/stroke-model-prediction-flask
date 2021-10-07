## Inputs
# age (years) blood_pressure (mm/Hg)
# blood_glucose_random (mgs/dl) blood_urea (mgs/dl)
# serum_creatinine (mgs/dl) hemoglobin (gms)
# white_blood_cells (cells/cumm) red_blood_cells (cells/cumm)
# hypertension (1 - yes, 0 -no) diabetes (1 - yes, 0 -no)
# * appetite(1 - good, 0 - poor)

## Imports
from pickle import load
import numpy as np

with open("models/CKD.pkl", "rb") as f:
    description, model, scaler = load(f)

# sample=
# {
# age,
# blood_pressure,
# blood_glucose_random,
# blood_urea,
# serum_creatinine,
# hemoglobin,
# white_blood_cells,
# red_blood_cells,
# hypertension,
# diabetes,
# appetite
# }

def preprocess_data(pred_dict):
    sample = np.array(list(pred_dict.values()))
    sample = sample.reshape(1, -1)
    standard_X_without_cat_feat = scaler.transform(sample[:, :-3])
    standard_X = np.concatenate((standard_X_without_cat_feat, sample[:, -3:]), axis=1)
    return standard_X

def make_kidney_prediction(data):
    # run prediction
    df = preprocess_data(data)
    pred_prob = model.predict_proba(df)
    # fetch predictions
    non_stroke, stroke = pred_prob[0]
    result = round((stroke * 100), 2)
    feature_impartances = model.feature_importances_
    return str(result), feature_impartances