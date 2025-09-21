```bash
#!/bin/bash

# Usage:
# ./upload_kaggle_to_gcs.sh <KAGGLE_URL> <FOLDER_NAME> [BUCKET_NAME]

# --- Input Parameters ---
KAGGLE_URL=$1
FOLDER_NAME=$2
BUCKET_NAME=${3:-sudss-main}  # Default bucket if not passed

# --- Validate Inputs ---
if [[ -z "$KAGGLE_URL" || -z "$FOLDER_NAME" ]]; then
  echo "Usage: $0 <KAGGLE_URL> <FOLDER_NAME> [BUCKET_NAME]"
  exit 1
fi

# --- Create GCS bucket if it doesn't exist ---
if ! gsutil ls -b gs://$BUCKET_NAME >/dev/null 2>&1; then
  echo "Creating bucket: gs://$BUCKET_NAME"
  gcloud storage buckets create gs://$BUCKET_NAME --location=us
else
  echo "Bucket gs://$BUCKET_NAME already exists."
fi

# --- Create a temp ZIP filename based on folder name ---
ZIP_FILE=~/"$FOLDER_NAME.zip"

echo "Downloading dataset from: $KAGGLE_URL"
curl -L -o "$ZIP_FILE" "$KAGGLE_URL"

# --- Unzip dataset ---
mkdir -p "$FOLDER_NAME"
unzip -q "$ZIP_FILE" -d "$FOLDER_NAME"

# --- Upload to GCS ---
echo "Uploading files to GCS: gs://$BUCKET_NAME/$FOLDER_NAME/"
gcloud storage cp -r "$FOLDER_NAME"/* "gs://$BUCKET_NAME/$FOLDER_NAME/"

echo "✅ Done!"

```

#### EHR downloaded to Gcloud
* https://synthea.mitre.org/downloads
    > https://academic.oup.com/jamia/article/25/3/230/4098271?login=false
    > https://github.com/synthetichealth/synthea/wiki/Basic-Setup-and-Running

* Open EMR for CDSS DB 
    > https://www.open-emr.org/
    > https://demo.openemr.io/openemr/interface/login/login.php?site=default

* Kaggle Data Sets
    * PMC - https://www.kaggle.com/datasets/priyamchoksi/pmc-patients-dataset-for-clinical-decision-support
    * CXIU -  https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
    * MIMIC-CXR - https://www.kaggle.com/datasets/nikeshreddypatlolla/mimic-cxr-dataset

