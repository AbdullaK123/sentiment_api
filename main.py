from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import app_settings
from app import PredictionRequest, PredictionResponse, get_inference_service, get_model, InferenceService
from transformers import Pipeline
from loguru import logger
from contextlib import asynccontextmanager
import gc
import torch
import time

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Loading model...")

    app.state.model: Pipeline = get_model() # pyright: ignore[reportInvalidTypeForm]

    logger.info("Configuring inference service...")
    app.state.inference_service: InferenceService = get_inference_service(app.state.model) # pyright: ignore[reportInvalidTypeForm]

    logger.info("Server is ready!")

    yield

    logger.info("🧹 Cleaning up model and services from memory...")

    del app.state.model 
    del app.state.inference_service
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ Cleanup completed!")

api = FastAPI(
    title="Sentiment Analysis API",
    description="An api for predicting sentiment in text",
    version="1.0",
    lifespan=lifespan
)

origins = app_settings.allowed_origins.split(',')
methods = app_settings.allowed_methods.split(',')
headers = app_settings.allowed_headers.split(',')

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,  
    allow_headers=headers,  
)

@api.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

@api.post('/v1/predict', response_model=PredictionResponse)
async def predict(
    request: Request,
    texts: PredictionRequest
) -> PredictionResponse:
    inference_service = request.app.state.inference_service
    return inference_service.predict(texts)


@api.get('/health')
async def health_check():
    return {"status": "healthy", "model_loaded": hasattr(api.state, 'model')}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)