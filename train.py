import os
import zipfile
import subprocess
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Download dataset if not present
if not os.path.exists("vgsales.csv"):
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "gregorut/videogamesales",
        "--unzip"
    ], check=True)

# Load
df = pd.read_csv("vgsales.csv").dropna()

# Target: hit if global sales > 1M copies
df["hit"] = (df["Global_Sales"] > 1.0).astype(int)

# Features
features = ["Platform", "Genre", "Publisher", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
df = df[features + ["hit"]]

# Encode categoricals
encoders = {}
for col in ["Platform", "Genre", "Publisher"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[features]
y = df["hit"]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
print(f"F1-score : {f1_score(y_test, preds):.4f}")

# Save model + encoders
joblib.dump({"model": model, "encoders": encoders, "features": features}, "model.pkl")
print("Model saved to model.pkl")
