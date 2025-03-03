from fastapi import APIRouter
from src.schemas.service_output import GetClassesOutput

router = APIRouter(tags=["Info"])


@router.get("/get_classes_info",
            summary="Получить информацию о классах.",
            description="Получить информацию о классах.",
            )
async def get_classes_info() -> GetClassesOutput:
    return GetClassesOutput()
