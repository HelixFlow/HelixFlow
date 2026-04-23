FROM python:3.10-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl for healthcheck / debug; bookworm ships most tooling already
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Layer-cache: install dependencies first
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip==24.3.1 \
    && pip install -r /app/requirements.txt

# Then copy source
COPY . /app/

# Non-root user
RUN useradd --create-home --shell /bin/bash helix \
    && chown -R helix:helix /app
USER helix

EXPOSE 11110

CMD ["uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "11110"]
