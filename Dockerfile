# TeamDocs AI - single-container image (API + static frontend).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    DATA_DIR=/data \
    MODEL_CACHE_DIR=/data/models \
    FRONTEND_DIR=/app/frontend

# libmagic1: file-type detection used by `unstructured` (DOCX/PPTX).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data && chown app:app /data

WORKDIR /app

# Dependencies first (cached layer). CPU-only torch keeps the image ~2 GB smaller than the CUDA wheel.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Official GitHub MCP server binary: lets the GitHub agent run inside the container without Docker-in-Docker.
ARG GITHUB_MCP_VERSION=v2.2.0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && arch="$(uname -m | sed 's/aarch64/arm64/')" \
    && curl -fsSL "https://github.com/github/github-mcp-server/releases/download/${GITHUB_MCP_VERSION}/github-mcp-server_Linux_${arch}.tar.gz" \
       | tar -xz -C /usr/local/bin github-mcp-server \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
ENV GITHUB_MCP_COMMAND=/usr/local/bin/github-mcp-server

COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend
RUN pip install --no-deps .

USER app
VOLUME ["/data"]
EXPOSE 8000

# /ready is 503 until the embedding model + vector store are loaded (first start downloads the model).
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/ready' % os.environ.get('APP_PORT','8000'), timeout=4)"

CMD ["python", "-m", "teamdocs.main"]
