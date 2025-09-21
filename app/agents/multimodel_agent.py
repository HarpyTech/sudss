"""
agents/multimodel_agent.py

MedicalImageModel — singleton wrapper around a Gemma3 multimodal model + processor.
Designed for FastAPI / background service use (no Streamlit calls). Uses robust
input filtering, device movement, and ensures text/image alignment by inserting
image special tokens when an image is present.

Replace MED_GEMMA_4B in config.constants with your actual model id string.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    Gemma3ImageProcessor,
)

from config.constants import MED_GEMMA_4B, LOG_FORMAT


# Module-level logger (container-friendly)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
__name__ = "MedicalImageModel"
logger = logging.getLogger(__name__)


class MedicalImageModel:
    """
    Singleton wrapper for a Gemma3 multimodal image+text model.

    Key behaviors:
    - Lazy singleton initialization so repeated imports reuse the loaded model.
    - Moves tensors to the model device robustly.
    - Filters out unwanted processor/metadata keys (e.g., 'num_crops') before calling generate().
    - Ensures text contains the expected number of image special tokens (adds token if needed).
    """

    __name__ = "MedicalImageModel"

    _instance: Optional["MedicalImageModel"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "MedicalImageModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, use_quantization: bool = False):
        if not self.__class__._initialized:
            self.model_id = MED_GEMMA_4B
            self.use_quantization = use_quantization
            self._initialize_model()
            self.__class__._initialized = True

    def _initialize_model(self) -> None:
        """Load processor, tokenizer and model. Keep logging for observability."""
        logger.info("Loading MedGemma model %s ...", self.model_id)

        # Load processor + tokenizer + model
        self.processor = Gemma3ImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

        if self.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        # Load the model (this can be heavy)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)

        logger.info("MedGemma model loaded successfully")

    # -------------------------
    # Utilities
    # -------------------------
    def _move_tensors_to_device(
        self, data_dict: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """
        Move any torch.Tensor values in data_dict to the specified device.
        Returns a new dict mapping the same keys to moved values \
        (or original values if not tensors).
        """
        out: Dict[str, Any] = {}
        for k, v in data_dict.items():
            try:
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device)
                elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                    out[k] = type(v)([t.to(device) for t in v])
                else:
                    # Some HF BatchEncodings and other objects implement .to(device)
                    if hasattr(v, "to") and not isinstance(v, (str, bytes)):
                        try:
                            out[k] = v.to(device)
                        except Exception:
                            out[k] = v
                    else:
                        out[k] = v
            except Exception:
                # Fallback to original value if anything goes wrong
                out[k] = v
        return out

    def _filter_generate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only keys that are accepted by model.generate / model forward.
        This prevents passing metadata keys like 'num_crops' which the transformers lib rejects.
        """
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
            logger.warning("Removed unexpected keys before model.generate(): %s", removed)
        return filtered

    def _find_or_add_image_token(
        self, candidates: Optional[Tuple[str, ...]] = None
    ) -> Tuple[str, int, bool]:
        """
        Find an existing image special token in the tokenizer or add a default one.
        Returns (token_str, token_id, added_flag).
        """
        if candidates is None:
            candidates = ("<image>", "<img>", "<image_0>", "<image0>", "[IMAGE]")

        # Search for an existing token
        for tok in candidates:
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tok_id = None
            if tok_id and tok_id != self.tokenizer.unk_token_id:
                logger.info("Found existing image token '%s' (id=%s) in tokenizer.", tok, tok_id)
                return tok, int(tok_id), False

        # Not found — add a new special token '<image>'
        image_token = "<image>"
        logger.info("Adding image special token '%s' to tokenizer.", image_token)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})

        # Resize model embeddings to accommodate new tokens (best effort)
        try:
            self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            logger.warning(
                "Failed to resize model token embeddings after adding special token: %s", e
            )

        image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        logger.info("Added image token '%s' with id=%s", image_token, image_token_id)
        return image_token, int(image_token_id), True

    def _infer_image_token_count(self, inputs: Dict[str, Any]) -> int:
        """
        Try multiple strategies to infer how many special image tokens the model expects.
        Returns an integer >= 1.
        Strategies (in order):
          1) Inspect model.config for fields like num_image_tokens
          or vision_config.num_image_tokens.
          2) If pixel_values present, run encoder (if available) and
          inspect last_hidden_state length.
          3) Heuristic from pixel_values shape (patch estimation).
          4) Fallback to 1.
        """
        cfg = getattr(self.model, "config", None)

        # Strategy 1: model config hints
        try:
            if cfg is not None:
                # Common config attributes
                if hasattr(cfg, "num_image_tokens") and cfg.num_image_tokens:
                    logger.info("Using config.num_image_tokens = %s", cfg.num_image_tokens)
                    return int(cfg.num_image_tokens)
                vision_cfg = getattr(cfg, "vision_config", None)
                if vision_cfg:
                    if hasattr(vision_cfg, "num_image_tokens") and vision_cfg.num_image_tokens:
                        logger.info(
                            "Using vision_config.num_image_tokens = %s", vision_cfg.num_image_tokens
                        )
                        return int(vision_cfg.num_image_tokens)
                    if hasattr(vision_cfg, "image_token_length") and vision_cfg.image_token_length:
                        logger.info(
                            "Using vision_config.image_token_length = %s",
                            vision_cfg.image_token_length,
                        )
                        return int(vision_cfg.image_token_length)
        except Exception:
            logger.debug(
                "Model config inspection failed for image token count inference", exc_info=True
            )

        # Strategy 2: run encoder to infer output length (best-effort)
        try:
            if "pixel_values" in inputs:
                pixel_values = inputs["pixel_values"].to(self.model.device)
                # Many models expose get_encoder() or encoder
                encoder = getattr(self.model, "get_encoder", None)
                if callable(encoder):
                    encoder = self.model.get_encoder()
                elif hasattr(self.model, "encoder"):
                    encoder = self.model.encoder
                else:
                    encoder = None

                if encoder is not None:
                    with torch.no_grad():
                        enc_out = encoder(pixel_values, return_dict=True)
                        last = getattr(enc_out, "last_hidden_state", None)
                        if last is not None:
                            token_count = int(last.shape[1])
                            logger.info(
                                "Inferred image token count from encoder outputs: %s", token_count
                            )
                            return max(1, token_count)
        except Exception:
            logger.debug("Encoder-run image token inference failed", exc_info=True)

        # Strategy 3: heuristic from pixel_values shape (patch estimation)
        try:
            pv = inputs.get("pixel_values", None)
            if pv is not None:
                shape = tuple(pv.shape)  # e.g., (batch, channels, H, W) or (batch, seq_len, dim)
                if len(shape) == 4:
                    _, c, h, w = shape
                    # crude patch size heuristic (patch ~16 px)
                    patches_est = max(1, (h // 16) * (w // 16))
                    logger.info(
                        "Estimated %s image tokens from pixel_values shape %s", patches_est, shape
                    )
                    return int(patches_est)
                elif len(shape) == 3:
                    # maybe already a patch sequence: (batch, seq_len, dim)
                    seq_len = shape[1]
                    logger.info(
                        "Using pixel_values sequence length %s as image token count", seq_len
                    )
                    return int(seq_len)
        except Exception:
            logger.debug("Pixel-values heuristic failed", exc_info=True)

        # Final fallback
        logger.warning("Could not infer image token count; defaulting to 1.")
        return 1

    # -------------------------
    # Core API
    # -------------------------
    def run_inference(
        self, image_path: Optional[str] = None, prompt: str = "Summarize the MRI scan"
    ) -> str:
        """
        Run multimodal inference.
        If image_path is provided, ensures pixel_values are passed and that the textual prompt
        contains the expected number of image special tokens so image embeddings align.
        Returns generated text (string).
        """
        logger.info("Running MedGemma inference... (image_path=%s)", bool(image_path))

        # 1) Prepare inputs
        if image_path:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = self._move_tensors_to_device(inputs, self.model.device)

            tokenized = self.tokenizer(prompt, return_tensors="pt")
            tokenized = self._move_tensors_to_device(tokenized, self.model.device)

            # Keep pixel_values from processor, add input_ids from tokenizer
            inputs["input_ids"] = tokenized.get("input_ids")
            # Optionally preserve any decoder-specific masks if needed in future
        else:
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            inputs = self._move_tensors_to_device(tokenized, self.model.device)

        # 2) If image present, ensure text contains image tokens aligning with image embeddings
        if "pixel_values" in inputs:
            image_token_str, image_token_id, _added = self._find_or_add_image_token()
            img_token_count = self._infer_image_token_count(inputs)

            input_ids = inputs.get("input_ids")
            if input_ids is None:
                # Build input_ids consisting only of image tokens
                inputs["input_ids"] = torch.tensor(
                    [[image_token_id] * img_token_count], device=self.model.device
                )
                logger.info("Created input_ids consisting of %d image tokens.", img_token_count)
            else:
                try:
                    if isinstance(input_ids, torch.Tensor):
                        img_tokens = torch.tensor(
                            [[image_token_id] * img_token_count], device=self.model.device
                        )
                        inputs["input_ids"] = torch.cat([input_ids, img_tokens], dim=1)
                        logger.info(
                            "Appended %d image tokens to input_ids (shape %s).",
                            img_token_count,
                            inputs["input_ids"].shape,
                        )
                    else:
                        # Coerce non-tensor to tensor
                        inputs["input_ids"] = torch.tensor(
                            [[image_token_id] * img_token_count], device=self.model.device
                        )
                        logger.info(
                            "Replaced non-tensor input_ids with %d image tokens.", img_token_count
                        )
                except Exception as e:
                    logger.warning(
                        """Failed to append image tokens to input_ids (%s).
                        Overwriting input_ids with image tokens.""",
                        e,
                    )
                    inputs["input_ids"] = torch.tensor(
                        [[image_token_id] * img_token_count], device=self.model.device
                    )

        # 3) Filter out unsupported keys before generation to avoid ValueError
        gen_inputs = self._filter_generate_inputs(inputs)

        # 4) Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**gen_inputs, max_new_tokens=512)

        # 5) Decode
        try:
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            generated_text = [
                self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
            ][0]

        logger.info("MedGemma inference completed.")
        return generated_text
