import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

file = './models/gb_1.bin'

model = joblib.load(file)

### Test Predictions
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
sample = [['Male', 67, 0, 1, 'Yes', 'Private', 'Urban', 228.69, 36.6, 'formerly smoked']] ## id 9046
sample_1 = [['Female', 61, 0, 0, 'Yes', 'Self-employed', 'Rural', 202.21, 0,'never smoked']]  ##id 51676
sample_2 = [['Male', 80, 0, 1, 'Yes', 'Private', 'Rural', 105.92,  32.5, 'never smoked']] ## id 31112
predictions_df = pd.DataFrame(sample_2, columns=columns)

## Read data
stroke_data = pd.read_csv('./input/Stroke.csv')
## Feature Engineering
# Age group - (old: 60+, young: 25-60, child: <25)
stroke_data['age_group'] = stroke_data['age'].apply(lambda row: 'old' if row>60 else ('young' if row>25 else 'child'))
predictions_df['age_group'] = predictions_df['age'].apply(lambda row: 'old' if row>60 else ('young' if row>25 else 'child'))
# Diabetes level - (normal: <140, diabetic: >200, prediabetic: 140-199)
stroke_data['diabetic'] = stroke_data['avg_glucose_level'].apply(lambda row: 'diabetic' if row>200 else ('normal' if row<140 else 'prediabetic'))
predictions_df['diabetic'] = predictions_df['avg_glucose_level'].apply(lambda row: 'diabetic' if row>200 else ('normal' if row<140 else 'prediabetic'))
# bmi levels - (underweight: <18.5, normal: 18.5-24,9, overweight: >25)
stroke_data['bmi_level'] = stroke_data['bmi'].apply(lambda row: 'underweight' if row<18.5 else ('normal' if row<25 else 'overweight'))
# predictions_df['bmi_level'] = predictions_df['bmi'].apply(lambda row: 'underweight' if row<18.5 else ('normal' if row<25 else 'overweight'))

print("Featured engineered data")
print(predictions_df.head())

## Handling categorical columns
cat_cols = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status',
    'age_group',
    'diabetic',
    'bmi_level'
]
for col in cat_cols:
    le = LabelEncoder()
    stroke_data[col] = le.fit_transform(stroke_data[col])
    predictions_df[col] = le.transform((predictions_df[col]))

print("Categorical encoded")
print(predictions_df.head())

## Handle missing values
null_col = ['bmi'] # only bmi has null values
# lets replace with mean as we dont see any outliers
for col in null_col:
    predictions_df[col] = predictions_df[col].fillna(stroke_data[col].mean())

## Feature scaling
num_cols = ['age', 'avg_glucose_level', 'bmi']
# for col in num_cols:
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stroke_data[num_cols].values)
scaled_pred_data = scaler.transform(predictions_df[num_cols].values)
print("Feature scaled")
df = pd.DataFrame(scaled_pred_data, columns=num_cols)
for col in num_cols:
    predictions_df[col] = df[col].values

pred = model.predict_proba(predictions_df)
print(f"Predicted Stroke Probability: {round((pred[0][0] * 100), 2)}%")