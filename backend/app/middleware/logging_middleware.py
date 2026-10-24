import uuid
from fastapi import Request
import structlog

logger = structlog.get_logger(__name__)


async def structlog_middleware(request: Request, call_next):
    """
    Middleware to inject a unique request_id into every HTTP request context.
    """
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    response = await call_next(request)
    response.headers["x-reqeust-id"] = request_id

    return response
