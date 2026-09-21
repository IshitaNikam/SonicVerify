from fastapi import APIRouter

from app import config
from app.services.detection_service import get_model_status

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": config.APP_NAME,
        "model_status": get_model_status(),
    }
