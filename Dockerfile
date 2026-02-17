# BirdNET on Raspberry Pi OS (Debian Bookworm)
# Build for Pi: docker build --platform linux/arm64 -t birdnet .
# Build for x86: docker build -t birdnet .
ARG TARGETPLATFORM
FROM --platform=$TARGETPLATFORM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Bookworm has Python 3.11 as default; install pip, venv, ffmpeg + runtime deps for birdnet/librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils \
    i2c-tools \
    python3-smbus \
    fonts-freefont-ttf \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libpng-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /home/pi/bird-project/logs

ENTRYPOINT ["python3", "main.py"]
