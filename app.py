from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


model = joblib.load("breast_cancer_model.pkl")

app = FastAPI(
    title="Breast Cancer Prediction, either M or B"
)
class Breast_Cancer_Data(BaseModel):
    radius_mean: float
    perimeter_mean: float
    area_mean: float
    concavity_mean: float
    concave_points_mean: float
    radius_worst: float
    perimeter_worst: float
    area_worst: float
    concavity_worst: float
    concave_points_worst: float 

@app.post("/predict")
def predict(data: Breast_Cancer_Data):
    input_data = np.array([[
        data.radius_mean, 
        data.perimeter_mean, 
        data.area_mean, 
        data.concavity_mean, 
        data.concave_points_mean,
        data.radius_worst, 
        data.perimeter_worst,
        data.area_worst,
        data.concavity_worst,
        data.concave_points_worst
    ]])
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    return {"prediction": float(prediction),
            "result": "Malignant" if prediction == 1 else "Benign"
            }