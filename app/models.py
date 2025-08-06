from sqlmodel import SQLModel
from typing import List

class PredictionRequest(SQLModel):
    texts: List[str]

class Prediction(SQLModel):
    text: str
    sentiment: str
    confidence: float

class PredictionResponse(SQLModel):
    predictions: List[Prediction]