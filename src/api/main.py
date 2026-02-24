"""
src/api/main.py — FastAPI Application Entry Point
===================================================

LAYMAN EXPLANATION:
    This is the "main reception desk" of the web server.
    It creates the FastAPI application, registers all the endpoints
    (from routes.py), and handles startup/shutdown tasks.

    When you run `python run.py --mode serve`, this file starts
    a web server that listens for loan scoring requests.

    Any system (bank website, mobile app, internal tool) can send
    a request to this server over HTTP and get a credit score back.

DATA SCIENTIST EXPLANATION:
    FastAPI app with:
    - Startup event: Load models into memory on server start
    - CORS middleware: Allow cross-origin requests (for web clients)
    - Request logging middleware: Log every request for observability
    - Structured error handling: Consistent error response format
    - Auto-generated API documentation at /docs (Swagger UI)
    - OpenAPI schema at /openapi.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import cfg
from src.api.routes import router
from src.models.predict import model_store


# ============================================================
# LIFESPAN — Startup and Shutdown Logic
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown.

    LAYMAN: When the server starts, we pre-load the ML models into memory
    so the first request doesn't have to wait for model loading.
    When the server stops, we clean up resources.

    This is like a restaurant opening — you prep everything BEFORE
    customers arrive, not when the first customer walks in.

    DATA SCIENTIST NOTE:
    Model loading takes ~1-2 seconds. Without lifespan management,
    the first request would time out or be very slow.
    The async context manager ensures clean shutdown (important for
    Kubernetes graceful termination).
    """
    # --- STARTUP ---
    logger.info("=" * 60)
    logger.info(f"Starting {cfg.api.api_title} v{cfg.api.api_version}")
    logger.info("=" * 60)

    # Attempt to pre-load models
    try:
        model_store.load()
        logger.info("Models pre-loaded successfully. API is ready.")
    except FileNotFoundError:
        logger.warning(
            "Model artifacts not found at startup. "
            "Run 'python run.py --mode train' first. "
            "API will still start but /score endpoints will return 503."
        )
    except Exception as e:
        logger.error(f"Failed to pre-load models: {e}")

    yield  # Application runs here

    # --- SHUTDOWN ---
    logger.info("Shutting down API server...")


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=cfg.api.api_title,
    description=cfg.api.api_description,
    version=cfg.api.api_version,
    lifespan=lifespan,
    docs_url="/docs",         # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",       # ReDoc UI at http://localhost:8000/redoc
    openapi_url="/openapi.json",
)


# ============================================================
# MIDDLEWARE
# ============================================================

# CORS (Cross-Origin Resource Sharing)
# LAYMAN: Allows web browsers from different websites to use this API.
# Without this, a bank's website at "bank.com" couldn't call this API
# running on "api.bank.com" due to browser security restrictions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log every incoming request and its response time.

    LAYMAN: Like a receptionist writing down who visited and how long
    they waited. This helps identify slow requests and debug issues.

    DATA SCIENTIST NOTE:
    Structured logging for observability. In production, you'd ship
    these logs to a centralized system (Datadog, CloudWatch, ELK stack)
    and create dashboards/alerts on:
    - p50/p95/p99 response times
    - Error rates
    - Requests per second
    """
    start_time = time.time()

    # Process the request
    response = await call_next(request)

    # Log after response
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration_ms:.1f}ms)"
    )

    return response


# ============================================================
# GLOBAL ERROR HANDLERS
# ============================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError (e.g., invalid input values)."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Invalid input", "detail": str(exc), "status_code": 422}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Check server logs.",
            "status_code": 500
        }
    )


# ============================================================
# INCLUDE ROUTES
# ============================================================

# Mount all routes from routes.py under the /api/v1 prefix
# Example: /score becomes /api/v1/score
app.include_router(router, prefix="/api/v1")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """
    LAYMAN: The homepage of the API. Shows basic information
    and links to documentation.
    """
    return {
        "message": f"Welcome to the {cfg.api.api_title}",
        "version": cfg.api.api_version,
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "scoring_endpoint": "/api/v1/score",
        "demo_endpoint": "/api/v1/demo",
    }


# ============================================================
# DIRECT RUN (for development)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {cfg.api.host}:{cfg.api.port}")
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=cfg.api.debug,  # Auto-reload on code changes (dev only)
        log_level="info",
    )
