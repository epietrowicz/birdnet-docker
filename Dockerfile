# BirdNET on Raspberry Pi OS (Debian Bookworm)
# Build for Pi: docker build --platform linux/arm64 -t birdnet .
# Build for x86: docker build -t birdnet .
ARG TARGETPLATFORM
FROM --platform=$TARGETPLATFORM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Bookworm has Python 3.11 as default; install pip, venv, ffmpeg + runtime deps for birdnet/librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir --break-system-packages \
    "numpy<2.0" \
    tflite-runtime \
    librosa \
    resampy \
    birdnetlib

ENTRYPOINT ["python3", "main.py"]
