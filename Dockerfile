# ──────────────────────────────────────────────────────────────────────
# AIDD Agent Backend — Multi-stage Conda build
#
# Build:  docker build -t aidd-backend:latest .
# Run:    docker run -p 8899:8899 --env-file .env aidd-backend:latest
# ──────────────────────────────────────────────────────────────────────
FROM continuumio/miniconda3:latest AS builder

WORKDIR /build

# Install conda environment from spec (this is the slowest layer, cached first)
COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda clean -afy \
    && find /opt/conda/envs/aidd-agent -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# ── Production stage ─────────────────────────────────────────────────
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy the pre-built conda environment
COPY --from=builder /opt/conda/envs/aidd-agent /opt/conda/envs/aidd-agent
ENV PATH="/opt/conda/envs/aidd-agent/bin:$PATH"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY app/ ./app/
COPY run.py .

# Do NOT copy .env — config is injected via K8s ConfigMap / Secret
ENV APP_ENV=prod
EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8899/health || exit 1

# Production mode: no hot-reload
CMD ["python", "run.py", "--no-reload"]
