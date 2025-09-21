import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")