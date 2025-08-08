FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . .

RUN uv pip install --no-cache-dir --system torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --no-cache-dir --system \
    transformers \
    fastapi \
    sqlmodel \
    uvicorn \
    loguru \
    pydantic-settings

EXPOSE 8000

CMD ["python", "main.py"]