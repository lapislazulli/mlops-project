import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl")

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]

class PredictRequest(BaseModel):
    features: list[float]  # expects 4 values: sepal_length, sepal_width, petal_length, petal_width

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    X = np.array(request.features).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()
    return {
        "prediction": int(pred),
        "class": IRIS_CLASSES[pred],
        "probabilities": proba
    }
