from utils.ehr_ingest import query_similar_records, load_faiss_index

load_faiss_index()

class RetrievalAgent:
    def retrieve(self, query: str, k: int = 3):
        return query_similar_records(query, top_k=k)
