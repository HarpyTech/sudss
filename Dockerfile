FROM python:3.11-alpine

WORKDIR /app

# Install system dependencies for Tesseract
# RUN apt-get update && apt-get install -y \
#     tesseract-ocr \
#     poppler-utils \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY .env .env

CMD ["streamlit", "run", "app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]
