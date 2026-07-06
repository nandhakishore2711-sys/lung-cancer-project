# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps needed by Pillow / matplotlib / opencv-less image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy the rest of the app
COPY . .

# Writable dirs used at runtime (uploaded scans, trained model weights)
RUN mkdir -p static/uploads saved_models

# Which backend to load at runtime — override at `docker run -e MODEL_BACKEND=pytorch`
ENV MODEL_BACKEND=tensorflow
ENV MODEL_ARCHITECTURE=custom
ENV PORT=8000

EXPOSE 8000

# 1 worker by default: image classification models are memory-heavy;
# raise --workers only if your host has the RAM to spare.
CMD gunicorn wsgi:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120
