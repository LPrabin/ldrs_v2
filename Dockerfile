# ============================================================================
# LDRS v2 — Living Document RAG System
#
# Multi-stage Docker build with three targets:
#   - api       : FastAPI server on port 8001
#   - streamlit : Streamlit web UI on port 8501
#   - cli       : One-shot CLI query runner
#
# Build:
#   docker build -t ldrs-v2 .                       # default target = api
#   docker build -t ldrs-v2-ui --target streamlit .  # Streamlit UI
#
# Run:
#   docker run --env-file .env -p 8001:8001 ldrs-v2
#   docker run --env-file .env -p 8501:8501 ldrs-v2-ui
#   docker run --env-file .env ldrs-v2-cli --query "What is EMD?"
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base — shared dependencies
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# System deps required by PyMuPDF (fitz) and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (layer cached as long as requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ldrs/         ldrs/
COPY rag/          rag/
COPY pageindex/    pageindex/
COPY scripts/      scripts/
COPY ldrs_v2_api.py .
COPY streamlit_app.py .

# Copy test data (PDFs + structure JSONs) so the default config works
# out of the box. For production, mount your own data via -v.
COPY tests/results/ tests/results/
COPY tests/pdfs/    tests/pdfs/

# Ensure project root is on PYTHONPATH
ENV PYTHONPATH=/app

# ---------------------------------------------------------------------------
# Stage 2: API server (default)
# ---------------------------------------------------------------------------
FROM base AS api

EXPOSE 8001

# Configurable via env vars (see ldrs_v2_api.py)
ENV LDRS_RESULTS_DIR=tests/results \
    LDRS_PDF_DIR=tests/pdfs \
    LDRS_MODEL=qwen3-vl \
    LDRS_PORT=8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "ldrs_v2_api.py"]

# ---------------------------------------------------------------------------
# Stage 3: Streamlit UI
# ---------------------------------------------------------------------------
FROM base AS streamlit

EXPOSE 8501

ENV LDRS_RESULTS_DIR=tests/results \
    LDRS_PDF_DIR=tests/pdfs \
    LDRS_MODEL=qwen3-vl

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# ---------------------------------------------------------------------------
# Stage 4: CLI runner
# ---------------------------------------------------------------------------
FROM base AS cli

ENV LDRS_RESULTS_DIR=tests/results \
    LDRS_PDF_DIR=tests/pdfs \
    LDRS_MODEL=qwen3-vl

ENTRYPOINT ["python", "scripts/run_ldrs_query.py"]
CMD ["--help"]
