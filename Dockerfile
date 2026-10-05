# syntax=docker/dockerfile:1

# ---- build: resolve dependencies with uv into a self-contained virtualenv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
# EXTRAS="ner" (default) adds GLiNER + CPU torch for names, addresses and organizations.
# EXTRAS="" builds a small patterns-only image (set PII_SHIELD_NER_ENABLED=false when running it).
ARG EXTRAS="ner"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)

# ---- runtime: slim image, non-root user ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app results ./results
COPY --chown=app:app configs ./configs
COPY --chown=app:app data ./data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    LOG_FORMAT=json

USER app
# Volume mount points must exist (owned by the app user) before Docker initializes named volumes.
RUN mkdir -p /home/app/.cache/huggingface /home/app/audit

# The NER model (~660 MB, PyTorch weights only) is downloaded on first start into $HF_HOME (mount a volume to
# keep it). BAKE_NER_MODEL=true downloads it at build time instead, for air-gapped or autoscaled deployments.
ARG BAKE_NER_MODEL=false
ARG NER_MODEL=knowledgator/gliner-pii-base-v1.0
RUN if [ "$BAKE_NER_MODEL" = "true" ]; then pii-shield download-model "$NER_MODEL"; fi

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "pii_shield.gateway.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
