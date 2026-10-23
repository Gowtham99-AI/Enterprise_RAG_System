
from fastapi import FastAPI

from enterprise_rag_platform.api.routes.answers import router as answers_router
from enterprise_rag_platform.api.routes.documents import router as documents_router
from enterprise_rag_platform.api.routes.health import router as health_router
from enterprise_rag_platform.api.routes.search import router as search_router
from enterprise_rag_platform.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
    )

    app.include_router(
        health_router,
        prefix=settings.api_prefix,
    )

    app.include_router(documents_router, prefix=settings.api_prefix)
    app.include_router(search_router, prefix=settings.api_prefix)
    app.include_router(answers_router, prefix=settings.api_prefix)
    return app


app = create_app()
