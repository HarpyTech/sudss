# Use slim Python 3.11 base
FROM python:3.11-slim

# -------------------- Set working directory -------------------- #
WORKDIR /app

# -------------------- Install system dependencies -------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libffi-dev \
    git \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------- Upgrade pip and setuptools -------------------- #
RUN python -m pip install --upgrade pip setuptools wheel

# -------------------- Copy requirements and install -------------------- #
COPY requirements.txt /app/requirements.txt

# Install CPU PyTorch first (kept in its own layer for better caching)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# -------------------- Install Gemma 3 compatible Transformers (if required) -------------------- #
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# -------------------- Copy application code -------------------- #
COPY ./app /app

# Ensure logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# -------------------- Expose FastAPI port -------------------- #
EXPOSE 8000

# -------------------- Run Uvicorn -------------------- #
# Entrypoint is app/app.py → module: app.app:app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
