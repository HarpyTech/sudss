Here's a full **end-to-end Python-based Agentic AI chatbot** system tailored for **Clinical Diagnosis Support**, designed for multimodal input (text, image, or lab reports), including routing agents, integration with BioGPT/PubMedGPT, summarization via Gemini, and deployment to **GCP Cloud Run** using **Docker** and **GitHub Actions**.

---

## 🧠 Overview Architecture

```
User Input (Text / Image / PDF)
    ↓
Extractor Agent (Text/Image/Lab Extractor)
    ↓
Router Agent
    → BioGPT Agent
    → PubMedGPT Agent
    ↓
Summarizer Agent (Gemini Flash/Gemma)
    ↓
✅ Doctor Summary Report
```

---

## 📦 Directory Structure

```
clinical_diagnosis_agent/
│
├── app/
│   ├── main.py
│   ├── agents/
│   │   ├── router_agent.py
│   │   ├── extractor_agent.py
│   │   ├── biogpt_agent.py
│   │   ├── pubmedgpt_agent.py
│   │   └── summarizer_agent.py
│   └── utils/
│       ├── image_parser.py
│       ├── document_parser.py
│       └── text_cleaner.py
│
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── deploy.yaml
```

---

## 🧪 Sample Agents and Main App (Python FastAPI)

### `app/main.py`

```python
from fastapi import FastAPI, UploadFile, File
from app.agents.router_agent import route_input
from app.agents.summarizer_agent import generate_summary

app = FastAPI()

@app.post("/diagnose/")
async def diagnose(input_file: UploadFile = File(...)):
    raw_data = await input_file.read()
    predictions = await route_input(raw_data, input_file.filename)
    summary = generate_summary(predictions)
    return {"summary_report": summary}
```

---

### `router_agent.py`

```python
from app.agents.extractor_agent import extract_content
from app.agents.biogpt_agent import get_biogpt_prediction
from app.agents.pubmedgpt_agent import get_pubmedgpt_prediction

async def route_input(raw_data, filename):
    content = await extract_content(raw_data, filename)
    biogpt_result = get_biogpt_prediction(content)
    pubmed_result = get_pubmedgpt_prediction(content)
    return {"biogpt": biogpt_result, "pubmedgpt": pubmed_result}
```

---

### `extractor_agent.py`

```python
from app.utils.image_parser import extract_text_from_image
from app.utils.document_parser import extract_text_from_pdf
from app.utils.text_cleaner import clean_text

async def extract_content(raw_data, filename):
    if filename.endswith(".pdf"):
        return clean_text(extract_text_from_pdf(raw_data))
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return clean_text(extract_text_from_image(raw_data))
    else:
        return clean_text(raw_data.decode("utf-8"))
```

---

### `biogpt_agent.py` (Placeholder for real inference)

```python
def get_biogpt_prediction(text: str):
    # Integrate BioGPT or call your hosted model
    return f"BioGPT Prediction for: {text[:100]}"
```

---

### `pubmedgpt_agent.py`

```python
def get_pubmedgpt_prediction(text: str):
    # Integrate PubMedGPT
    return f"PubMedGPT Prediction for: {text[:100]}"
```

---

### `summarizer_agent.py` (Gemini/Gemma Placeholder)

```python
def generate_summary(predictions: dict):
    combined = f"BioGPT: {predictions['biogpt']}\n\nPubMedGPT: {predictions['pubmedgpt']}"
    # Call Gemini Flash / Gemma here
    return f"Summary based on Gemini: {combined[:300]}"
```

---

## 🐳 Dockerfile

```Dockerfile
FROM python:3.10

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./app ./app
COPY main.py .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 🧾 `requirements.txt`

```
fastapi
uvicorn
pytesseract
pdfplumber
Pillow
```

---

## 🚀 GitHub Actions CI/CD for GCP Cloud Run

### `.github/workflows/deploy.yaml`

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Authenticate with GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: '${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}'

      - name: Set up gcloud
        uses: google-github-actions/setup-gcloud@v2

      - name: Build Docker image
        run: |
          gcloud builds submit --tag gcr.io/$PROJECT_ID/clinical-agent-app

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy clinical-agent-app \
            --image gcr.io/$PROJECT_ID/clinical-agent-app \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated
```

### 🔐 GitHub Secrets Required

* `GCP_SERVICE_ACCOUNT_KEY` → Your base64-encoded `service-account.json`
* `PROJECT_ID` → Your GCP project ID

---

## ✅ Deployment Checklist

1. Enable Cloud Run & Cloud Build API
2. Create Service Account with Cloud Run Admin, Storage Admin, Artifact Registry, and Build permissions
3. Add secrets to GitHub
4. Push to `main` to auto-deploy

---

Would you like me to generate a **zip file** with this structure, or code to integrate actual APIs (BioGPT hosted, Gemini model calls)?
