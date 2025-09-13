import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import streamlit as st

from config.constants import MED_GEMMA_4B


class MedicalImageModel:
    """
    Singleton wrapper for the Med-Gemma-4B model from Hugging Face.
    Ensures the model is loaded only once.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedicalImageModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize_model()
            self.__class__._initialized = True

    def _initialize_model(self):
        print(f"🚀 Loading model {MED_GEMMA_4B}...")
        self.processor = AutoProcessor.from_pretrained(MED_GEMMA_4B)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MED_GEMMA_4B, torch_dtype=torch.float16, device_map="auto"
        )
        print(f"✅ Model {MED_GEMMA_4B} loaded once")

    def run_inference(self, image_path: str, prompt: str = "Summarize the MRI scan"):
        """
        Run inference on a given medical image.
        """

        st.info("Running MedGemma inference...")
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        st.info("MedGemma inference completed.")
        return generated_text
