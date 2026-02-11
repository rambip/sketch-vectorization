FROM ghcr.io/astral-sh/uv:debian-slim
RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
