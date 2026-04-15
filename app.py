import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
artifact = joblib.load("model.pkl")
model = artifact["model"]
encoders = artifact["encoders"]
features = artifact["features"]

class PredictRequest(BaseModel):
    Platform: str
    Genre: str
    Publisher: str
    NA_Sales: float
    EU_Sales: float
    JP_Sales: float
    Other_Sales: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    data = request.model_dump()
    for col in ["Platform", "Genre", "Publisher"]:
        le = encoders[col]
        val = data[col]
        if val not in le.classes_:
            return {"error": f"Unknown {col}: {val}"}
        data[col] = le.transform([val])[0]

    df = pd.DataFrame([data])[features]
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].tolist()
    return {
        "prediction": int(pred),
        "label": "hit" if pred == 1 else "flop",
        "confidence": round(max(proba), 4)
    }
