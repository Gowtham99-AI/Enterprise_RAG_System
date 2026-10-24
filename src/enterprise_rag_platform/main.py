
from contextlib import asynccontextmanager

from fastapi import FastAPI

from enterprise_rag_platform.api.exception_handlers import (
    unhandled_exception_handler,
)
from enterprise_rag_platform.api.middleware import CorrelationIdMiddleware
from enterprise_rag_platform.api.routes.answers import router as answers_router
from enterprise_rag_platform.api.routes.documents import (
    router as documents_router,
)
from enterprise_rag_platform.api.routes.health import router as health_router
from enterprise_rag_platform.api.routes.search import router as search_router
from enterprise_rag_platform.config import get_settings
from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.rag import RAGService
from enterprise_rag_platform.services.reranking import RerankingService

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_service = EmbeddingService()
    reranking_service = RerankingService()

    app.state.embedding_service = embedding_service
    app.state.reranking_service = reranking_service
    app.state.rag_service = RAGService(
        embedding_service=embedding_service,
    )

    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

app.add_middleware(CorrelationIdMiddleware)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(search_router, prefix=settings.api_prefix)
app.include_router(answers_router, prefix=settings.api_prefix)
