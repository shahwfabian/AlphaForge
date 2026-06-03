# ─────────────────────────────────────────────────────────────────────────────
# AlphaForge — Multi-Stage Docker Build
# ─────────────────────────────────────────────────────────────────────────────
#
# Stage 1 (builder): install Python dependencies into a virtualenv.
# Stage 2 (runtime): copy only the venv + application source. No build tools.
#
# Usage
# ─────
#   Build:   docker build -t alphaforge:latest .
#   Run:     docker run -p 8501:8501 alphaforge:latest
#   Compose: docker compose up
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System-level build dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel

# Create isolated virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="AlphaForge" \
      org.opencontainers.image.description="Modular Quantitative Trading Research Platform" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="AlphaForge Contributors"

WORKDIR /app

# Copy compiled virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/raw /app/data/processed /app/exports \
    && chown -R appuser:appuser /app

# Copy application source (in dependency order)
COPY --chown=appuser:appuser src/           ./src/
COPY --chown=appuser:appuser tests/         ./tests/
COPY --chown=appuser:appuser docs/          ./docs/
COPY --chown=appuser:appuser .streamlit/    ./.streamlit/
COPY --chown=appuser:appuser app.py         ./app.py
COPY --chown=appuser:appuser requirements.txt ./requirements.txt

USER appuser

# Streamlit port
EXPOSE 8501

# Health check — polls the Streamlit health endpoint
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1

# Default command
ENTRYPOINT ["streamlit", "run", "app.py"]
CMD [ \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false" \
]
