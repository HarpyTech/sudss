import torch
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma3ImageProcessor,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from PIL import Image
import streamlit as st

from config.constants import MED_GEMMA_4B


class MedicalImageModel:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, use_quantization=False):
        if not self._initialized:
            self.model_id = MED_GEMMA_4B
            self.use_quantization = use_quantization
            self._initialize_model()
            self.__class__._initialized = True

    def _initialize_model(self):
        st.info(f"🚀 Loading MedGemma model {self.model_id} ...")
        self.processor = Gemma3ImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

        if self.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
        st.success("✅ MedGemma model loaded successfully")

    def run_inference(self, image_path: str = None, prompt: str = "Summarize the MRI scan"):
        st.info("🧠 Running MedGemma inference...")

        if image_path:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            inputs["input_ids"] = (
                self.tokenizer(prompt, return_tensors="pt").to(self.model.device).input_ids
            )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        st.info("✅ MedGemma inference completed.")
        return generated_text
