
from contextlib import asynccontextmanager

from fastapi import FastAPI

from enterprise_rag_platform.services.embedding import EmbeddingService
from enterprise_rag_platform.services.rag import RAGService
from enterprise_rag_platform.services.reranking import RerankingService

# Keep your existing router and middleware imports here.
# Keep your existing router registrations and middleware configuration.


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_service = EmbeddingService()
    reranking_service = RerankingService()

    app.state.embedding_service = embedding_service
    app.state.reranking_service = reranking_service

    app.state.rag_service = RAGService(
        embedding_service=embedding_service
    )

    yield


app = FastAPI(
    title="Enterprise RAG Platform",
    lifespan=lifespan,
)

# IMPORTANT:
# Restore your existing middleware, exception handlers, and router
# registrations below this point.
