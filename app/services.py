from transformers import pipeline, Pipeline
from app.models import PredictionRequest, Prediction, PredictionResponse
from app.config import app_settings
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import HTTPException, status
from app.config import app_settings

def get_model(
    task: str = app_settings.task,
    model_name: str = app_settings.model
) -> Pipeline:
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )

    return pipeline(
        task=task, 
        model=quantized_model, 
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

class InferenceService:

    def __init__(self, model: Pipeline):
        self.model = model
        self.min_text_length = int(app_settings.min_text_length)
        self.max_text_length = int(app_settings.max_text_length)

    def _validate_text(self, text: str) -> bool:
        if len(text) < self.min_text_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"One of your texts is too short. All must be at least {self.min_text_length} characters."
            )
        if len(text) > self.max_text_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"One of your texts is too long. All must be less than {self.max_text_length} characters"
            )
        return True

    def _get_prediction(self, text: str) -> Prediction:
        if self._validate_text(text):
            prediction = self.model(text)[0]
            return Prediction(
                text=text,
                sentiment=prediction['label'],
                confidence=prediction['score']
            )
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        return PredictionResponse(
            predictions= [
                self._get_prediction(text)
                for text in request.texts
            ]
        )
    

def get_inference_service(
    model: Pipeline
) -> InferenceService:
    return InferenceService(model)
