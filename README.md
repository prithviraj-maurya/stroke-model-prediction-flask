### This repository contains api for getting stroke prediction.
### The api recives the following parameters
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
### The output would be the prediction percentage of Stroke.
### Several pretrained models are present in the models folder and input data in the input.