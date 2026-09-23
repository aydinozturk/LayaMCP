# syntax=docker/dockerfile:1.7
# Laya MCP server with the english + multilingual checkpoints baked in (offline start).
#
#   CPU  (amd64 + arm64) -> aydinozturk/laya-mcp:latest
#     docker buildx build --platform linux/amd64,linux/arm64 -t aydinozturk/laya-mcp:latest --push .
#
#   NVIDIA GPU (amd64, CUDA 12.6 wheels, needs host driver >= 525) -> aydinozturk/laya-mcp:cuda
#     docker buildx build --platform linux/amd64 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 -t aydinozturk/laya-mcp:cuda --push .
#
#   Other options:
#     --build-arg LAYA_BAKE_MODELS=english,multilingual,typed-decisions   # bake all three
#     --build-arg LAYA_BAKE_MODELS= --build-arg HF_HUB_OFFLINE=0          # slim, download on first use

FROM python:3.12-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.0
# TORCH_DISABLE_NATIVE_JIT: torch >= 2.14 routes some CUDA ops (e.g. the bmm in ModernBERT's
# rotary embedding) to Triton kernels that JIT-compile a C helper on first use, which fails
# in a slim image without a compiler. Disabling it keeps those ops on stock cuBLAS.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_DISABLE_NATIVE_JIT=1

# CUDA builds also get a C compiler, so anything else that reaches Triton can still JIT.
RUN case "$TORCH_INDEX" in \
      */whl/cu*) apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
                 && rm -rf /var/lib/apt/lists/* ;; \
    esac

RUN pip install --index-url ${TORCH_INDEX} torch==${TORCH_VERSION}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Pin the laya release the image was tested with; upstream ships often.
ARG LAYA_VERSION=0.3.9
RUN pip install . "laya==${LAYA_VERSION}"

RUN useradd --create-home --uid 1000 laya && mkdir -p /opt/hf && chown laya /opt/hf
USER laya

# Download the checkpoints once at build time (as the runtime user, so no chown layer
# duplicates the weights) and containers start offline.
ARG LAYA_BAKE_MODELS=english,multilingual
RUN if [ -n "$LAYA_BAKE_MODELS" ]; then \
      python -c "import os; from laya import Router; Router().preload([m for m in os.environ['LAYA_BAKE_MODELS'].split(',') if m])"; \
    fi

# Baked images resolve checkpoints from the local cache only (no network at startup). Build
# with --build-arg HF_HUB_OFFLINE=0 for a slim image, or pass -e HF_HUB_OFFLINE=0 at runtime
# to fetch a checkpoint that was not baked (e.g. typed-decisions).
ARG HF_HUB_OFFLINE=1
ENV HF_HUB_OFFLINE=${HF_HUB_OFFLINE} \
    MCP_TRANSPORT=stdio \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    LAYA_MODELS=english,multilingual
EXPOSE 8000

ENTRYPOINT ["laya-mcp"]
