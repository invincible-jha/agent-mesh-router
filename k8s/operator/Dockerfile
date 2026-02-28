FROM python:3.12-slim

# Security: run as non-root user
RUN groupadd --gid 1000 operator \
 && useradd --uid 1000 --gid operator --shell /bin/bash --create-home operator

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy operator source
COPY main.py .
COPY controllers/ ./controllers/

# Ensure the controllers package is importable
RUN touch controllers/__init__.py 2>/dev/null || true

USER operator

# kopf uses SIGTERM for graceful shutdown — no custom signal handler needed
ENTRYPOINT ["python", "-m", "kopf", "run", "--liveness=http://0.0.0.0:8080/healthz", "main.py"]

# Liveness probe port (exposed by kopf built-in health endpoint)
EXPOSE 8080
