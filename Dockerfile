FROM debian:trixie-slim

# --- Build-time options ------------------------------------------------
# SUPERTONIC_REF : branch/tag to clone from the upstream repo
# WITH_RU        : set to "true" to install the extra Russian auto-stress
#                  dependencies (adds build time + image size)
ARG SUPERTONIC_REF=main
ARG WITH_RU=false

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src

COPY ./ ./

# pyproject.toml already pins the runtime deps (supertonic, wyoming,
# sentence-stream, num2words, onnxruntime, numpy). The optional "ru" extra
# pulls in the extra libraries needed for Russian auto-stress.
RUN if [ "${WITH_RU}" = "true" ]; then \
        pip install --no-cache-dir ".[ru]"; \
    else \
        pip install --no-cache-dir .; \
    fi

RUN apt-get autoremove --purge -y git build-essential

RUN chmod +x ./docker_run.sh && mkdir -p /data

# Default Wyoming port used by wyoming_supertonic
EXPOSE 10209

ENTRYPOINT ["./docker_run.sh"]
