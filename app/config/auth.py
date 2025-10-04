from huggingface_hub import login

from app.config.variables import HF_TOKEN


def hf_login():
    """
    Authenticate with Hugging Face. Uses HF_TOKEN from env.
    """
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN not found. Please set it in .env (local) or Secret Manager (GCP)."
        )

    login(token=HF_TOKEN)
    print("✅ Hugging Face login successful")
