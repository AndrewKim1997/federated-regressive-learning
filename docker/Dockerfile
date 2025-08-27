# docker/Dockerfile
# Minimal, reproducible CPU image for federated-regressive-learning
# - Python 3.10 (matches env/environment.yml)
# - Optional: install PyTorch CPU wheels
# - Non-root user for safety

FROM python:3.10-slim

# ---- OS deps (slim-friendly) ----
# libgomp1: scikit-learn OpenMP runtime
# git: optional, handy inside container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 git \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- Non-root user ----
ARG USER=appuser
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} ${USER} && \
    adduser --disabled-password --gecos "" --uid ${UID} --gid ${GID} ${USER}
WORKDIR /app

# ---- Build-time toggles ----
# INSTALL_TORCH=1 installs torch/torchvision/torchaudio (CPU wheels)
# TORCH_INDEX can be switched to a CUDA index if you build a GPU image later
ARG INSTALL_TORCH=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG INSTALL_DEV=0  # 1 = install dev extras (pytest, black, etc.)

# ---- Install project (deps via pyproject.toml) ----
# Copy just the metadata first to leverage Docker layer cache
COPY pyproject.toml README.md ./
RUN python -m pip install --upgrade pip
RUN if [ "$INSTALL_DEV" = "1" ]; then \
      pip install -e .[dev]; \
    else \
      pip install -e .; \
    fi

# ---- Optional: PyTorch (CPU) ----
RUN if [ "$INSTALL_TORCH" = "1" ]; then \
      pip install torch torchvision torchaudio --index-url ${TORCH_INDEX}; \
    fi

# ---- Copy the rest of the repo ----
# (After deps to maximize caching)
COPY . .

# ---- Permissions ----
RUN chown -R ${USER}:${USER} /app
USER ${USER}

# ---- Default command ----
# Tip: override this with `docker run ... python scripts/run_federated.py ...`
CMD ["python", "-c", "import frl; print('FRL package import OK')"]
