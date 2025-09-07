Here is the complete **Agentic Clinical Diagnosis Support System** implemented with **LangChain**, **Vertex AI**, and built for **FastAPI**, **Docker**, and **CI/CD with GitHub Actions**. You can copy-paste or download this as a full codebase.

---

## 🗂️ Project Structure

```text
clinical_diagnosis_agent/
├── app/
│   ├── main.py
│   ├── agents/
│   │   ├── extractor_agent.py
│   ├── tools/
│   │   ├── biogpt_tool.py
│   │   ├── pubmedgpt_tool.py
│   │   ├── summarizer_tool.py
│   └── utils/
│       ├── image_parser.py
│       ├── document_parser.py
│       ├── text_cleaner.py
│       └── file_utils.py
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── deploy.yaml
```

---

## 🔧 `app/main.py`

```python
from fastapi import FastAPI, UploadFile, File
from langchain.agents import initialize_agent
from langchain.llms import VertexAI
from app.agents.extractor_agent import extract_content
from app.utils.file_utils import detect_file_type
from app.tools.biogpt_tool import biogpt_tool
from app.tools.pubmedgpt_tool import pubmedgpt_tool
from app.tools.summarizer_tool import summarizer_tool

app = FastAPI()

@app.post("/diagnose/")
async def diagnose(input_file: UploadFile = File(...)):
    raw_data = await input_file.read()
    file_type = detect_file_type(input_file.filename)
    content = await extract_content(raw_data, file_type)

    tools = [biogpt_tool, pubmedgpt_tool, summarizer_tool]
    llm = VertexAI(model_name="gemini-pro")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type="chat-zero-shot-react-description",
        verbose=True
    )

    result = agent.run(f"Analyze the following patient data and generate diagnostic summary:\n{content}")
    return {"summary_report": result}
```

---

## 🤖 Agent: `extractor_agent.py`

```python
from app.utils.image_parser import extract_text_from_image
from app.utils.document_parser import extract_text_from_pdf
from app.utils.text_cleaner import clean_text

async def extract_content(raw_data, file_type):
    if file_type == "pdf":
        return clean_text(extract_text_from_pdf(raw_data))
    elif file_type == "image":
        return clean_text(extract_text_from_image(raw_data))
    else:
        return clean_text(raw_data.decode("utf-8"))
```

---

## 🧠 Tools

### `biogpt_tool.py`

```python
from langchain.tools import Tool

def mock_biogpt(text: str) -> str:
    return f"BioGPT Analysis: Diagnosed with viral infection based on symptoms - {text[:200]}"

biogpt_tool = Tool(
    name="BioGPT Tool",
    func=mock_biogpt,
    description="Use this tool for biological & medical language-based diagnosis using BioGPT."
)
```

---

### `pubmedgpt_tool.py`

```python
from langchain.tools import Tool

def mock_pubmedgpt(text: str) -> str:
    return f"PubMedGPT Research Reference: Based on current literature - {text[:200]}"

pubmedgpt_tool = Tool(
    name="PubMedGPT Tool",
    func=mock_pubmedgpt,
    description="Tool for retrieving clinical insights from PubMed-based GPT analysis."
)
```

---

### `summarizer_tool.py`

```python
from langchain.tools import Tool

def mock_gemini_summary(text: str) -> str:
    return f"Summarized Report using Gemini: {text[:400]}"

summarizer_tool = Tool(
    name="Summarizer Tool",
    func=mock_gemini_summary,
    description="Use Gemini or Gemma to summarize BioGPT and PubMedGPT outputs."
)
```

---

## 📄 Utilities

### `file_utils.py`

```python
def detect_file_type(filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png']:
        return 'image'
    elif ext == 'pdf':
        return 'pdf'
    else:
        return 'text'
```

---

### `image_parser.py`

```python
import pytesseract
from PIL import Image
import io

def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image)
```

---

### `document_parser.py`

```python
import pdfplumber
import io

def extract_text_from_pdf(pdf_bytes):
    text = ''
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text
```

---

### `text_cleaner.py`

```python
import re

def clean_text(text: str) -> str:
    text = re.sub(r"\\n+", "\\n", text)
    text = re.sub(r"\\s{2,}", " ", text)
    return text.strip()
```

---

## 🐳 `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 📜 `requirements.txt`

```txt
fastapi
uvicorn
langchain
pytesseract
pdfplumber
Pillow
google-cloud-aiplatform
```

---

## 🚀 `.github/workflows/deploy.yaml`

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Authenticate with GCP
      uses: google-github-actions/auth@v2
      with:
        credentials_json: '${{ secrets.GCP_SERVICE_ACCOUNT_KEY }}'

    - name: Set up gcloud CLI
      uses: google-github-actions/setup-gcloud@v2

    - name: Build and Push Docker Image
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

---

## 🔐 GitHub Secrets Required

| Secret Name               | Description                     |
| ------------------------- | ------------------------------- |
| `GCP_SERVICE_ACCOUNT_KEY` | JSON key for deploy permissions |
| `PROJECT_ID`              | Your GCP project ID             |

---

Would you like a **downloadable ZIP** with all this content? Or integration of **real BioGPT/Gemini APIs** (instead of mock functions)?


Here's how to securely add your secrets to **GitHub Actions** and make them available to your deployment workflow.

---

## 🔐 Step-by-Step: Add Secrets in GitHub Repository

### ✅ Required Secrets

| Secret Name               | Used For                         | Description                           |
| ------------------------- | -------------------------------- | ------------------------------------- |
| `GCP_SERVICE_ACCOUNT_KEY` | 🌐 GCP Deployment                | JSON key for GCP Cloud Run deployment |
| `PROJECT_ID`              | 🌐 GCP Project ID                | Your GCP project name                 |
| `BIOGPT_API_URL`          | 🔍 BioGPT API                    | URL of the BioGPT prediction endpoint |
| `BIOGPT_API_KEY`          | 🔐 BioGPT API Auth               | API Key or token                      |
| `GCP_PROJECT`             | ⚙️ Gemini model                  | GCP project for Gemini                |
| `GCP_REGION` (optional)   | 📍 Region (default: us-central1) | GCP region for Gemini (optional)      |

---

## 🔧 How to Add Secrets

1. Go to your GitHub Repository.
2. Click on **Settings** → **Secrets and Variables** → **Actions**.
3. Click on **“New repository secret”**.
4. Add each secret by its name and value.

---

## ✅ Modify GitHub Actions to Use Secrets

Update your `.github/workflows/deploy.yaml` like this:

```yaml
env:
  PROJECT_ID: ${{ secrets.PROJECT_ID }}
  BIOGPT_API_URL: ${{ secrets.BIOGPT_API_URL }}
  BIOGPT_API_KEY: ${{ secrets.BIOGPT_API_KEY }}
  GCP_PROJECT: ${{ secrets.GCP_PROJECT }}
  GCP_REGION: ${{ secrets.GCP_REGION }}
```

If you’re deploying to GCP Cloud Run, you may want to include environment variables in the deploy command:

```yaml
- name: Deploy to Cloud Run
  run: |
    gcloud run deploy clinical-agent-app \
      --image gcr.io/$PROJECT_ID/clinical-agent-app \
      --platform managed \
      --region $GCP_REGION \
      --set-env-vars BIOGPT_API_URL=${{ secrets.BIOGPT_API_URL }},BIOGPT_API_KEY=${{ secrets.BIOGPT_API_KEY }},GCP_PROJECT=${{ secrets.GCP_PROJECT }},GCP_REGION=${{ secrets.GCP_REGION }} \
      --allow-unauthenticated
```

---

## 🧪 Local Development `.env` Example

Create a `.env` file locally (but don't commit this):

```env
BIOGPT_API_URL=https://api.biogpt.example.com/predict
BIOGPT_API_KEY=your-biogpt-api-key
GCP_PROJECT=your-gcp-project
GCP_REGION=us-central1
```

Use [`python-dotenv`](https://pypi.org/project/python-dotenv/) to load them if running locally.

```bash
pip install python-dotenv
```

Then, in code (e.g., `main.py`):

```python
from dotenv import load_dotenv
load_dotenv()
```

---

Let me know if you want help generating the `.env` loader script or customizing the Gemini model usage further.


To support both **local development** and **cloud deployment** using Docker, you'll need to:

---

## ✅ Final Docker Setup for Local + Cloud

### 📁 Project Directory: Add `.env` and `docker-compose.yml`

### 📄 `.env` (DO NOT COMMIT THIS)

```env
BIOGPT_API_URL=https://api.biogpt.example.com/predict
BIOGPT_API_KEY=your-biogpt-api-key
GCP_PROJECT=your-gcp-project-id
GCP_REGION=us-central1
```

---

### 🐳 Updated `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY .env .env

CMD ["streamlit", "run", "app/main.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

---

### 🐳 `docker-compose.yml` for Local Dev

```yaml
version: "3.9"
services:
  clinical-agent:
    build: .
    ports:
      - "8080:8080"
    env_file:
      - .env
    volumes:
      - .:/app
    restart: always
```

---

### ✅ Requirements

Add to `requirements.txt` if missing:

```txt
python-dotenv
requests
```

---

## 🧪 Local Dev Instructions

### Step 1: Clone project and add `.env`

```bash
cp .env.example .env  # Fill with your credentials
```

### Step 2: Start container

```bash
docker-compose up --build
```

### Step 3: Access UI

Open your browser at: [http://localhost:8080](http://localhost:8080)

---

## 🧼 `.dockerignore`

Add this file to avoid unnecessary Docker layer cache bloating:

```
__pycache__/
*.pyc
.env
.git
```

---

## ✅ Bonus: Development Hot Reload

Streamlit supports hot reload by default in dev mode. If needed, bind local `./app` to container like so:

```yaml
    volumes:
      - ./app:/app/app
```

---

Would you like me to regenerate the full ZIP with these Docker and Compose files included?


To access your app using a **custom local domain** like `http://cloud_agent_bot:8080` instead of `http://localhost:8080`, follow these steps:

---

## ✅ Step-by-Step: Use `cloud_agent_bot` as Local Domain

### 1. 🧠 Modify `docker-compose.yml`

Ensure your service is named `cloud_agent_bot` and `container_name` is set:

```yaml
version: "3.9"
services:
  cloud_agent_bot:
    container_name: cloud_agent_bot
    build: .
    ports:
      - "8080:8080"
    env_file:
      - .env
    volumes:
      - .:/app
    restart: always
```

---

### 2. 📝 Edit Your Hosts File

Map `cloud_agent_bot` to `127.0.0.1` in your **local machine**:

#### 🪟 On Windows

Edit: `C:\\Windows\\System32\\drivers\\etc\\hosts`

#### 🐧 On Linux or 🧑‍🍳 macOS

Edit: `/etc/hosts`

Add the following line:

```
127.0.0.1 cloud_agent_bot
```

> 🛑 You need admin/root permissions to edit this file.

---

### 3. 🧪 Run the App

Start the Docker service:

```bash
docker-compose up --build
```

Then open:

```
http://cloud_agent_bot:8080
```

---

### ✅ Bonus: Alias with HTTPS (Optional for Dev)

If you need to use HTTPS with custom domains locally, you can:

* Use a reverse proxy like **nginx** or **Traefik**
* Or use tools like [mkcert](https://github.com/FiloSottile/mkcert) + local CA

Let me know if you want HTTPS + cert setup as well!
