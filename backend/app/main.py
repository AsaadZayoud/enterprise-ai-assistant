# backend/app/main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from backend.app.core.config import settings
from backend.app.routers import rag, documents  # Import the new router
from backend.app.core.logging import get_logger, setup_logging
from backend.app.middleware.logging_middleware import structlog_middleware
from dotenv import load_dotenv
from langfuse import get_client


load_dotenv()


# 1. Initialize structured logging configuration
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🟢 Startup
    logger.info("application_startup", project_name=settings.PROJECT_NAME, debug=settings.DEBUG)
    langfuse_client = get_client()
    
    yield 
    
    # 🔴 Shutdown
    try:
        langfuse_client.flush()
        logger.info("langfuse_traces_flushed_successfully")
    except Exception as e:
        logger.error("langfuse_flush_failed", error=str(e))
        
    logger.info("application_shutdown")


# 3. Initialize FastAPI App with Lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# 4. Register Middleware
app.middleware("http")(structlog_middleware)

# 5. Include API Routers
app.include_router(documents.router)
app.include_router(rag.router)

@app.get("/")
async def root():
    logger.info("root_endpoint_accessed")
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running successfully!"}