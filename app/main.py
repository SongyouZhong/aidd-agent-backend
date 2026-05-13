"""FastAPI application entry point."""

from __future__ import annotations

import warnings
from contextlib import asynccontextmanager

# Suppress a spurious structlog UserWarning that fires when langgraph_api's
# processor chain includes both format_exc_info and structlog's own exception
# renderer.  The warning is cosmetic — exception info is still captured.
warnings.filterwarnings(
    "ignore",
    message="Remove `format_exc_info` from your processor chain",
    category=UserWarning,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth as auth_router
from app.api import chat as chat_router
from app.api import events as events_router
from app.api import files as files_router
from app.api import messages as messages_router
from app.api import projects as projects_router
from app.api import sessions as sessions_router
from app.api import targets as targets_router
from app.api import tasks as tasks_router
from app.api import traces as traces_router
from app.core.config import settings
from app.services.background_runner import background_runner
from app.services.task_registry import reap_stale_tasks
from app.storage.redis_client import close_redis, get_redis
from app.storage.s3 import s3_storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly init singletons so failures surface at startup, not first request.
    await s3_storage.start()
    await get_redis()
    # Mark any tasks that were in-flight when the process last died as failed.
    await reap_stale_tasks()
    try:
        yield
    finally:
        await background_runner.shutdown()
        await s3_storage.stop()
        await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Type", "Cache-Control"],
    )

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.APP_ENV}

    app.include_router(auth_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(projects_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(sessions_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(messages_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(chat_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(files_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(targets_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(traces_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(tasks_router.router, prefix=settings.API_V1_PREFIX)
    app.include_router(events_router.router, prefix=settings.API_V1_PREFIX)

    return app


app = create_app()
