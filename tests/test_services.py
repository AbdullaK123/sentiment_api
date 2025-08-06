from .conftest import test_request, test_service
from app.services import InferenceService
from app.models import PredictionRequest

def test_inference_service(test_service: InferenceService, test_request: PredictionRequest):

    result = test_service.predict(test_request)

    print(result)

    sentiments = [prediction.sentiment for prediction in result.predictions]

    assert sentiments == ["NEGATIVE", "POSITIVE"]