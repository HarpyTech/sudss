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
COPY requirements.txt .

# Install CPU PyTorch first to leverage caching
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -------------------- Install Gemma 3 compatible Transformers -------------------- #
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# -------------------- Copy app code -------------------- #
COPY ./app ./app

# -------------------- Expose Streamlit port -------------------- #
EXPOSE 8080

# -------------------- Run Streamlit -------------------- #
CMD ["streamlit", "run", "app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]
