from sqlmodel import SQLModel, Field
from pydantic import validator
from typing import List

class PredictionRequest(SQLModel):
    texts: List[str] = Field(..., min_items=1, max_items=50, description="List of texts to analyze")
    
    @validator('texts')
    def validate_texts(cls, texts):
        from app.config import app_settings

        if not texts:
            raise ValueError(f"Must have at least one text!")
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")
            
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace only")
                
            if len(text) < app_settings.min_text_length:
                raise ValueError(f"Text at index {i} is too short (min {app_settings.min_text_length} chars)")
                
            if len(text) > app_settings.max_text_length:
                raise ValueError(f"Text at index {i} is too long (max {app_settings.max_text_length} chars)")
        
        return texts
    
class Prediction(SQLModel):
    text: str
    sentiment: str
    confidence: float

class PredictionResponse(SQLModel):
    predictions: List[Prediction]