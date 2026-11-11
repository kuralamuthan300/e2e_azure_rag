FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.13-slim

ENV PYTHONNOUSERSITE=1

# Apply Debian security updates so the runtime image does not ship known
# base-image vulnerabilities. The zip CLI is not used at runtime.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get purge -y zip \
    && rm -rf /var/lib/apt/lists/*

# Keep the base image's system Python packages at patched versions. The MCP
# server runs from its own virtual environment, but image scanners also see the
# system site-packages shipped by the Python base image.
RUN python -m pip install --no-cache-dir --upgrade \
    "setuptools>=78.1.1" \
    "msgpack>=1.2.3" \
    "urllib3>=2.8.0" \
    && rm -rf /root/.cache/pip

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

CMD ["/app/.venv/bin/python", "server.py"]
