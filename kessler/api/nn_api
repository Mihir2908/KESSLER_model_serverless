from fastapi import APIRouter
from typing import List
from kessler.nn import LSTMPredictor

router = APIRouter()

@router.post("/nn/load_model")
def load_model_api(model_path: str):
    global model
    model = LSTMPredictor.load(model_path)
    return {"status": "Model loaded successfully"}

@router.post("/nn/predict_event")
def predict_event_api(cdms: List[dict]):
    # Placeholder: implement event prediction logic
    return {"prediction": "Not implemented in this stub"}

@router.post("/nn/predict_next")
def predict_next_api(cdms: List[dict]):
    # Placeholder: implement next-step prediction logic
    return {"prediction": "Next CDM prediction (implement logic)"}