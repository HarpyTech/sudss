import os
import json
from datetime import datetime

# BigQuery client (only used in GCP mode)
from google.cloud import bigquery

class AuditLogger:
    def __init__(self, local_file="audit_logs.json"):
        self.local_file = local_file
        self.env = os.getenv("DEPLOY_ENV", "local")  # "local" or "gcp"

        if self.env == "gcp":
            self.bq_client = bigquery.Client()
            self.bq_dataset = os.getenv("BQ_DATASET")
            self.bq_table = os.getenv("BQ_TABLE")

    def log_event(self, action, user="system", metadata=None):
        """
        Log an audit event.
        :param action: str → action performed (e.g. 'run_inference')
        :param user: str → who triggered it
        :param metadata: dict → extra details
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "metadata": metadata or {}
        }

        if self.env == "local":
            self._log_local(event)
        elif self.env == "gcp":
            self._log_bigquery(event)

    def _log_local(self, event):
        """Append logs into JSON file locally."""
        logs = []
        if os.path.exists(self.local_file):
            with open(self.local_file, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []

        logs.append(event)

        with open(self.local_file, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"✅ Logged locally: {event}")

    def _log_bigquery(self, event):
        """Insert logs into BigQuery table."""
        table_id = f"{self.bq_client.project}.{self.bq_dataset}.{self.bq_table}"

        errors = self.bq_client.insert_rows_json(table_id, [event])
        if errors:
            print(f"❌ BigQuery insert errors: {errors}")
        else:
            print(f"✅ Logged to BigQuery: {event}")
