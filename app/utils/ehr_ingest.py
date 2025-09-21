import json
import os
from pathlib import Path
import faiss
import numpy as np
from utils.embeddings import generate_embeddings_batch
from config.config import MODE, FAISS_INDEX_PATH

if MODE == "cloud":
    from google.cloud import storage

DIM = 384  # embedding dimension (MiniLM)
faiss_index = faiss.IndexFlatL2(DIM)
id_map = []

def _load_local(path: str):
    with open(path, "r") as f:
        return json.load(f)

def _load_cloud(gcs_path: str):
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    return json.loads(data)

def ingest_fhir_file(fhir_path: str):
    """Ingest single FHIR JSON file"""
    global faiss_index, id_map
    data = _load_local(fhir_path) if MODE == "local" else _load_cloud(fhir_path)
    patient_id = data.get("id", "unknown")

    texts, metas = [], []
    for entry in data.get("entry", []):
        resource = entry.get("resource", {})
        r_type = resource.get("resourceType", "")
        text = ""

        if r_type == "DiagnosticReport":
            text = resource.get("conclusion", "")
        elif r_type == "Observation":
            text = f"{resource.get('code', {}).get('text','')}: {resource.get('valueQuantity', {}).get('value','')}"
        elif r_type == "Condition":
            text = resource.get("code", {}).get("text", "")

        if text:
            texts.append(text)
            metas.append({"patient_id": patient_id, "type": r_type, "text": text})

    if texts:
        embeddings = generate_embeddings_batch(texts).astype("float32")
        faiss_index.add(embeddings)
        id_map.extend(metas)

    save_faiss_index()
    return {"patient_id": patient_id, "records_added": len(texts)}

def save_faiss_index():
    if MODE == "local":
        Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
        with open(os.path.join(FAISS_INDEX_PATH, "metadata.json"), "w") as f:
            json.dump(id_map, f)
    else:
        tmp_index, tmp_meta = "/tmp/index.faiss", "/tmp/metadata.json"
        faiss.write_index(faiss_index, tmp_index)
        with open(tmp_meta, "w") as f:
            json.dump(id_map, f)

        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv("SUDSS_BUCKET"))
        bucket.blob("faiss_index/index.faiss").upload_from_filename(tmp_index)
        bucket.blob("faiss_index/metadata.json").upload_from_filename(tmp_meta)

def load_faiss_index():
    global faiss_index, id_map
    if MODE == "local":
        index_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        meta_path = os.path.join(FAISS_INDEX_PATH, "metadata.json")
        if os.path.exists(index_path):
            faiss_index = faiss.read_index(index_path)
            with open(meta_path, "r") as f:
                id_map = json.load(f)
    else:
        tmp_index, tmp_meta = "/tmp/index.faiss", "/tmp/metadata.json"
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv("SUDSS_BUCKET"))
        bucket.blob("faiss_index/index.faiss").download_to_filename(tmp_index)
        bucket.blob("faiss_index/metadata.json").download_to_filename(tmp_meta)
        faiss_index = faiss.read_index(tmp_index)
        with open(tmp_meta, "r") as f:
            id_map = json.load(f)

def query_similar_records(query: str, top_k: int = 3):
    query_emb = generate_embeddings_batch([query]).astype("float32")
    D, I = faiss_index.search(query_emb, top_k)
    results = [{"score": float(D[0][j]), **id_map[I[0][j]]} for j in range(len(I[0]))]
    return results
