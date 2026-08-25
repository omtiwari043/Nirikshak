FROM python:3.11-slim

# System deps:
#   libgl1/libglib2.0-0 -> opencv + paddle runtime
#   libgomp1            -> paddlepaddle's OpenMP runtime
#   libpq-dev/gcc       -> psycopg2 build
#   curl                -> healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/storage/uploads /app/storage/reports

ENV PYTHONUNBUFFERED=1 \
    OCR_LANG=en \
    OCR_USE_GPU=false \
    HOME=/root

# Pre-download the PaddleOCR detection/recognition/angle-classification models
# at build time so the running container never needs outbound internet access
# and the very first request in production isn't stuck downloading models.
# Models land in ~/.paddleocr and are baked into this image layer.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False, show_log=False)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# -w 1: PaddleOCR's model is loaded per-worker-process and is memory-heavy
# (several hundred MB); keep a single worker unless you've sized the host
# for more and verified memory headroom (see docs/DEPLOYMENT.md).
# --preload additionally shares the model load in gunicorn's master before
# forking (relevant if you do scale workers).
CMD ["gunicorn", "app.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--timeout", "120", "--preload"]
