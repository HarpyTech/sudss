import logging
import torch
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma3ImageProcessor,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from PIL import Image

from config.constants import MED_GEMMA_4B, LOG_FORMAT

# Configure logger for this module (container-friendly)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


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
        logger.info("Loading MedGemma model %s ...", self.model_id)

        self.processor = Gemma3ImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

        if self.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)

        logger.info("MedGemma model loaded successfully")

    def _move_tensors_to_device(self, data_dict, device):
        """
        Move any torch.Tensor values in data_dict to the specified device.
        Returns a new dict with same keys and mapped values.
        """
        out = {}
        for k, v in data_dict.items():
            try:
                # If v is a tensor, move it
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device)
                # If v is a list/tuple of tensors, move each
                elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                    out[k] = type(v)([t.to(device) for t in v])
                # Some processors return objects supporting .to(device)
                elif hasattr(v, "to") and not isinstance(v, (str, bytes)):
                    try:
                        out[k] = v.to(device)
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
            except Exception:
                # Failsafe: keep original value
                out[k] = v
        return out

    def _filter_generate_inputs(self, inputs: dict):
        """
        Keep only keys that are sensible for model.generate.
        This prevents passing accidental meta-keys like 'num_crops' which Transformers rejects.
        """
        # Common valid model/generation input keys (not exhaustive)
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "inputs_embeds",
            "encoder_outputs",
            "past_key_values",
            "pixel_values",
            "position_ids",
            "token_type_ids",
            "labels",
        }

        filtered = {k: v for k, v in inputs.items() if k in allowed_keys}
        removed = [k for k in inputs.keys() if k not in filtered]
        if removed:
            logger.warning("Removed unexpected keys before model.generate: %s", removed)
        return filtered

    def run_inference(self, image_path: str = None, prompt: str = "Summarize the MRI scan"):
        logger.info("Running MedGemma inference...")

        # Prepare inputs
        if image_path:
            image = Image.open(image_path).convert("RGB")
            # processor returns a dict (possibly with meta keys like 'num_crops')
            inputs = self.processor(images=image, return_tensors="pt")
            # Move processor outputs to model device
            inputs = self._move_tensors_to_device(inputs, self.model.device)

            # Tokenize prompt and move to device
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            tokenized = self._move_tensors_to_device(tokenized, self.model.device)

            # Merge tokenized text inputs into inputs (but avoid overwriting pixel_values etc.)
            inputs["input_ids"] = tokenized.get("input_ids")
        else:
            # Only textual prompt
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            inputs = self._move_tensors_to_device(tokenized, self.model.device)

        # Filter out unsupported keys (e.g., 'num_crops')
        gen_inputs = self._filter_generate_inputs(inputs)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**gen_inputs, max_new_tokens=512)

        # Decode generated ids to text using tokenizer
        try:
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            generated_text = [
                self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
            ][0]

        logger.info("MedGemma inference completed.")
        return generated_text
