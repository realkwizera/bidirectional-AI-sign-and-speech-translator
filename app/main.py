from fastapi import FastAPI, UploadFile, File
import torch

from services.predictor_service import PredictorService

app = FastAPI()
service = PredictorService()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    result = await service.predict(file)

    return {
        "prediction": result["label"],
        "confidence": result["confidence"]
    }