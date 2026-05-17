from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check():
    return HealthResponse(status="ok", version=settings.APP_VERSION)
