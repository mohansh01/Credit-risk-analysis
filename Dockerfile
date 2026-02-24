# ============================================================
# Dockerfile — Containerize the Credit Risk Scoring Engine
# ============================================================
#
# LAYMAN EXPLANATION:
#   A Docker container is like a self-contained box that includes
#   everything needed to run the application:
#   - Python runtime
#   - All dependencies (XGBoost, FastAPI, etc.)
#   - The application code
#   - Trained model files
#
#   Once "containerized," the app runs IDENTICALLY everywhere:
#   your laptop, a bank's server, AWS, Google Cloud, anywhere.
#   No "it works on my machine" problems.
#
# BUILD:
#   docker build -t credit-risk-engine .
#
# TRAIN MODEL FIRST (one-time):
#   docker run credit-risk-engine python run.py --mode train
#
# RUN API:
#   docker run -p 8000:8000 credit-risk-engine python run.py --mode serve
#
# THEN VISIT:
#   http://localhost:8000/docs
#
# DATA SCIENTIST EXPLANATION:
#   Multi-stage build not used here for simplicity.
#   For production, use multi-stage build to separate:
#   1. Build stage: install all deps including dev tools
#   2. Runtime stage: only production deps + app code (smaller image)
# ============================================================

# --- Base Image ---
# Python 3.11 slim = minimal Python install without extra OS packages
# "slim" reduces image size from ~900MB to ~200MB
FROM python:3.11-slim

# --- Labels ---
LABEL maintainer="credit-risk-team"
LABEL description="ML-powered credit risk scoring engine"
LABEL version="1.0.0"

# --- Environment Variables ---
# PYTHONDONTWRITEBYTECODE: Don't create .pyc files (saves space)
# PYTHONUNBUFFERED: Print logs immediately (no buffering)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# --- Working Directory ---
WORKDIR /app

# --- Install System Dependencies ---
# These are needed for some Python packages to compile
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/ removes package lists after install to save space

# --- Install Python Dependencies ---
# Copy requirements first (Docker caches this layer if requirements don't change)
# This means if you only change your code (not requirements), Docker won't
# reinstall packages — significantly speeds up rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# --no-cache-dir = don't cache downloaded packages (saves image size)

# --- Copy Application Code ---
COPY . .

# --- Create directories needed at runtime ---
RUN mkdir -p data/raw data/processed models/artifacts reports

# --- Expose Port ---
# Port 8000 is where the FastAPI server listens
# -p 8000:8000 in docker run maps container port to host port
EXPOSE 8000

# --- Health Check ---
# Docker will periodically check if the container is healthy
# by hitting the /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# --- Default Command ---
# This runs when you do: docker run credit-risk-engine
# Override with: docker run credit-risk-engine python run.py --mode train
CMD ["python", "run.py", "--mode", "serve"]
