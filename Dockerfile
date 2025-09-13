# Use Python 3.11 Alpine
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install system dependencies required to build C-extensions
RUN apk add --no-cache \
    build-base \
    python3-dev \
    musl-dev \
    linux-headers \
    libffi-dev \
    git \
    bash \
    curl \
    jpeg-dev \
    zlib-dev \
    tesseract-ocr \
    tesseract-ocr-data-eng

# Upgrade pip, setuptools, wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU wheels first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./app ./app

# Expose port and run Streamlit
EXPOSE 8080
CMD ["streamlit", "run", "app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]
