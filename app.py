from fastapi import FastAPI
import joblib
import numpy as np
from sklearn.datasets import load_iris

app = FastAPI()

iris = load_iris()

#Load the trained model
model = joblib.load("modeliris.joblib")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/")
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = iris.target_names[prediction][0]
    return {"class": class_name}