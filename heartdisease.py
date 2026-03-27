"""
Heart Disease Prediction System
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ─── Data Collection and Processing ───────────────────────────────────────────

# Loading the csv dataset into a pandas dataframe
heart_data = pd.read_csv('heart_disease_data.csv')

# Print first five rows
print(heart_data.head())

# Print last five rows
print(heart_data.tail())

# Check the shape of the data
print("Shape:", heart_data.shape)

# Getting some info about the dataset
heart_data.info()

# Checking for missing values
print("Missing values:\n", heart_data.isnull().sum())

# Statistical measures of the data
print(heart_data.describe())

# Checking the distribution of the target variable
print(heart_data['target'].value_counts())

"""
1 → Defective Heart (Heart Disease Present)
0 → Healthy Heart (No Heart Disease)
"""

# ─── Splitting Features and Target ────────────────────────────────────────────

X = heart_data.drop(columns=['target'])
Y = heart_data['target']

print("\nFeatures:\n", X)
print("\nTarget:\n", Y)

# ─── Splitting into Training and Test Data ─────────────────────────────────────

# FIX: removed the erroneous `train_test =` assignment
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print("\nShapes → Full:", X.shape, "| Train:", X_train.shape, "| Test:", X_test.shape)

# ─── Model Training ────────────────────────────────────────────────────────────

model = LogisticRegression(max_iter=1000)

# Training the logistic regression model with training data
model.fit(X_train, Y_train)

# ─── Model Evaluation ──────────────────────────────────────────────────────────

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('\nAccuracy on Training Data:', training_data_accuracy)

# Accuracy on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on Testing Data:', testing_data_accuracy)

# ─── Prediction System (FastAPI) ───────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: PatientData):
    # Map the JSON data to the model's required input tuple
    input_data = (data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal)
    input_array = np.asarray(input_data).reshape(1, -1)
    
    prediction = model.predict(input_array)[0]
    probs = model.predict_proba(input_array)[0]
    confidence = int(max(probs) * 100)
    
    # Generic key factors for the frontend
    factors = []
    if data.thal == 3: factors.append("Reversible defect")
    if data.cp >= 1: factors.append("Chest pain present")
    if data.ca > 0: factors.append("Major vessels affected")
    if not factors: factors.append("Based on overall profile")

    return {
        "prediction": int(prediction),
        "confidence": confidence,
        "keyFactors": factors[:3]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)