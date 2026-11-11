FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.13-slim

ENV PYTHONNOUSERSITE=1

# Apply Debian security updates so the runtime image does not ship known
# base-image vulnerabilities (zlib, zip, and related packages).
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

CMD ["/app/.venv/bin/python", "server.py"]
