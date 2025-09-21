import os

MODE = os.getenv("SUDSS_MODE", "local")   # "local" or "cloud"

if MODE == "local":
    FAISS_INDEX_PATH = "backend/utils/faiss_index"
    DATA_PATH = "backend/utils/patients"
else:
    GCS_BUCKET = os.getenv("SUDSS_BUCKET", "sudss-main")
    FAISS_INDEX_PATH = f"gs://{GCS_BUCKET}/faiss_index"
    DATA_PATH = f"gs://{GCS_BUCKET}/patients"
