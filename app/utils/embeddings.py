from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings_batch(texts):
    """Generate embeddings for a list of texts (batch mode)"""
    return np.array(_model.encode(texts, batch_size=16, show_progress_bar=False))
