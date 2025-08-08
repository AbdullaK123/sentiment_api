# Builder stage
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# copy uv binaries to image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# copy app code
COPY . .

# install dependancies with uv
RUN uv pip install --no-cache-dir --system torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --no-cache-dir --system \
    transformers \
    fastapi \
    sqlmodel \
    uvicorn \
    loguru \
    pydantic-settings

# remove artifacts leftover after installation
RUN find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2 >/dev/null || true  && \
    find /usr/local -name "*.pyo" -delete

# runtime stage
FROM python:3.12-slim-bookworm AS prod

# cleanly install curl
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# copy dependancies and binaries
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/


RUN groupadd -r appuser && useradd -r -g appuser -m appuser

RUN mkdir /home/appuser/.cache && chown -R appuser:appuser /home/appuser

WORKDIR /app

RUN chown appuser:appuser /app

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["python", "main.py"]