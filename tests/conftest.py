import pytest
from app.services import get_model, InferenceService
from app.models import PredictionRequest

@pytest.fixture
def test_service() -> InferenceService:
    model = get_model()
    return InferenceService(model)


@pytest.fixture
def test_request() -> PredictionRequest:
    return PredictionRequest(
        texts=[
            "This sucks! This is a horrible movie!",
            "OMG OMG I LOOOOOVE THIS! It's the best!"
        ]
    )


