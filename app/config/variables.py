import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
IMG_TOKENS_OVERRIDE = int(os.environ.get("IMG_TOKENS_OVERRIDE", default=256))
IMG_TOKENS_HEURISTIC_CAP = int(os.environ.get("IMG_TOKENS_HEURISTIC_CAP", default=1024))
IMG_TOKENS_MAX_APPEND = int(os.environ.get("IMG_TOKENS_MAX_APPEND", default=2048))
