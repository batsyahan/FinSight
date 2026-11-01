# FinSight API image:  docker build -t finsight .  &&  docker run -p 8000:8000 --env-file .env finsight
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# install dependencies first so code edits don't bust the dependency layer
COPY pyproject.toml README.md ./
COPY finsight/__init__.py finsight/__init__.py
RUN pip install ".[api,market,langgraph,rag,redis,ui,pdf]"

COPY finsight ./finsight
COPY app ./app
COPY examples ./examples
RUN pip install --no-deps . && useradd --create-home --uid 1000 finsight && mkdir -p /app/reports /app/.finsight_cache \
    && chown -R finsight /app
USER finsight

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "finsight.api:app", "--host", "0.0.0.0", "--port", "8000"]
