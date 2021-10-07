This repository contains api for getting stroke prediction.
The output would be the prediction percentage of Stroke.
Several pretrained models are present in the models folder and input data in the input.
### App is deployed on heroku on [link](https://stroke-model-prediction-flask.herokuapp.com).
API is /predict POST method with below payload

The api recives the following parameters
```json
{
    "gender":"Female",
    "age":45,
    "hypertension":1,
    "heart_disease":1,
    "ever_married":"Yes",
    "work_type":"Self-employed",
    "Residence_type":"Rural",
    "avg_glucose_level":160,
    "bmi":18,
    "smoking_status":"smokes"
}
```
Infor about the model

Given clinical parameters about a patient, the model will predict the probability whether or not a patient is likely to get stroke
When building the model, the AutoML library flaml was used.
.pkl file contains description, model and scaler.

Model itself for prediction (presence of stroke) model was buit on data of 5000 people.
its input:

Age (years) (age of the patient)

Avg_glucose_level (mg/dL) (avg_glucose_level: average glucose level in blood)

BMI: body mass index

Hypertension (Resting blood pressure is persistently at or above 140/90 mmHg)

All gathered features (except 'Hypertension' - it's binary feature) must be transformed by scaler before input

### Scores of the model:

AUC           0.9391839723109691

Accuracy      0.9394418114797262

Precision     0.9212362911266201

Recall        0.9625

F1-score      0.9414161996943454