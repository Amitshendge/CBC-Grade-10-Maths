# Use a full Ubuntu base for better audio library support
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed for ffmpeg, Pillow, psycopg2 and audio handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    build-essential \
    gcc \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libsndfile1 \
    libcairo2-dev \
    libsndfile1 \
    libasound2 \
    ca-certificates \
    pkg-config \
    libasound2 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install requirements
COPY ../../code_base/backend_api/requirements.txt ./
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY ../../code_base/backend_api ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "1800"]
