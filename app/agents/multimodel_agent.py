"""
agents/multimodel_agent.py

Complete MedicalImageModel singleton for Gemma3 multimodal usage.

Features:
- Loads Gemma3ImageProcessor, AutoTokenizer, Gemma3ForConditionalGeneration.
- Moves tensors to model device robustly.
- Filters out unwanted processor metadata keys (e.g., 'num_crops').
- Attempts to infer the correct number of image placeholder tokens by:
    1) environment override IMG_TOKENS_OVERRIDE
    2) model.config hints
    3) running the encoder (best effort)
    4) a capped patch-based heuristic
- Adds/finds an image special token if needed and resizes model embeddings.
- Appends the correct number of image tokens to input_ids (configurable behavior).
- Detailed logging for diagnostics (uses LOG_FORMAT from config.constants).
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
from config.variables import IMG_TOKENS_OVERRIDE, IMG_TOKENS_HEURISTIC_CAP, IMG_TOKENS_MAX_APPEND

# -------------------------
# Logging configuration
# -------------------------
# Use the user's requested logging block
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
__name__ = "MedicalImageModel"
logger = logging.getLogger(__name__)


class MedicalImageModel:
    """
    Singleton wrapper for Gemma3 multimodal inference.
    """

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
        """
        Load processor, tokenizer and model. This can be expensive.
        """
        logger.info("Loading MedGemma model %s ...", self.model_id)
        # Load processor and tokenizer
        self.processor = Gemma3ImageProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if self.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        # Load model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)

        logger.info("MedGemma model loaded successfully (id=%s)", self.model_id)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _move_tensors_to_device(
        self, data_dict: Dict[str, Any], device: torch.device
    ) -> Dict[str, Any]:
        """
        Move any torch.Tensor values in data_dict to the specified device.
        Returns a new dict mapping keys to moved values (or original values if not tensors).
        """
        out: Dict[str, Any] = {}
        for k, v in data_dict.items():
            try:
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device)
                elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], torch.Tensor):
                    out[k] = type(v)([t.to(device) for t in v])
                else:
                    # Some HF BatchEncodings support .to(device)
                    if hasattr(v, "to") and not isinstance(v, (str, bytes)):
                        try:
                            out[k] = v.to(device)
                        except Exception:
                            out[k] = v
                    else:
                        out[k] = v
            except Exception:
                out[k] = v
        return out

    def _filter_generate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only keys accepted by model.generate / forward to avoid transformer validate errors.
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
        Look for existing image special token(s) in tokenizer. If none found, add '<img>'.
        Returns (token_str, token_id, added_flag).
        """
        if candidates is None:
            candidates = ("<img>", "<image>", "<image_0>", "<image0>", "[IMAGE]")

        # Search known candidates
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        for tok in candidates:
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tok_id = None
            if tok_id is not None and unk_id is not None:
                if tok_id != unk_id:
                    logger.info(
                        "Found existing image token '%s' (id=%s) in tokenizer.", tok, tok_id
                    )
                    return tok, int(tok_id), False
            else:
                # If tokenizer has no unk_token_id, accept any non-zero id
                if tok_id:
                    logger.info(
                        "Found existing image token '%s' (id=%s) in tokenizer.", tok, tok_id
                    )
                    return tok, int(tok_id), False

        # Add default image token
        image_token = "<img>"
        logger.info(
            "No existing image token found. Adding special token '%s' to tokenizer.", image_token
        )
        try:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
            # resize model embeddings (best effort; some sharded setups may warn)
            self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            logger.warning("Failed to add/resize tokenizer/model embeddings for image token: %s", e)

        image_id = self.tokenizer.convert_tokens_to_ids(image_token)
        logger.info("Added image token '%s' with id=%s", image_token, image_id)
        return image_token, int(image_id), True

    def _infer_image_token_count(self, inputs: Dict[str, Any]) -> int:
        """
        Infer number of image placeholder tokens the model expects.

        Order of precedence:
          1) Environment override IMG_TOKENS_OVERRIDE
          2) Model config hints (num_image_tokens / vision_config.*)
          3) Run encoder and inspect last_hidden_state sequence length (best effort)
          4) Heuristic from pixel_values shape (capped)
          5) Fallback to 1
        """
        # 1) Env override
        override = IMG_TOKENS_OVERRIDE
        if override:
            try:
                val = int(override)
                logger.info("Using image token override from environment: %d", val)
                return max(1, val)
            except Exception:
                logger.warning("Invalid IMG_TOKENS_OVERRIDE value: %s", override)

        # 2) Model config hints
        cfg = getattr(self.model, "config", None)
        try:
            if cfg is not None:
                if getattr(cfg, "num_image_tokens", None):
                    val = int(cfg.num_image_tokens)
                    logger.info("Using config.num_image_tokens = %d", val)
                    return max(1, val)
                vision_cfg = getattr(cfg, "vision_config", None)
                if vision_cfg is not None:
                    if getattr(vision_cfg, "num_image_tokens", None):
                        val = int(vision_cfg.num_image_tokens)
                        logger.info("Using vision_config.num_image_tokens = %d", val)
                        return max(1, val)
                    if getattr(vision_cfg, "image_token_length", None):
                        val = int(vision_cfg.image_token_length)
                        logger.info("Using vision_config.image_token_length = %d", val)
                        return max(1, val)
        except Exception:
            logger.debug(
                "Model config inspection failed when inferring image token count", exc_info=True
            )

        # 3) Try running encoder to get hidden state length
        try:
            if "pixel_values" in inputs:
                pv = inputs["pixel_values"].to(self.model.device)
                encoder = None
                # get_encoder may be callable or attribute
                try:
                    enc_getter = getattr(self.model, "get_encoder", None)
                    if callable(enc_getter):
                        encoder = self.model.get_encoder()
                    elif hasattr(self.model, "encoder"):
                        encoder = self.model.encoder
                except Exception:
                    encoder = None

                if encoder is not None:
                    logger.info("Attempting to run encoder to infer image embedding length...")
                    with torch.no_grad():
                        enc_out = encoder(pv, return_dict=True)
                    last = getattr(enc_out, "last_hidden_state", None)
                    if last is not None:
                        token_count = int(last.shape[1])
                        logger.info(
                            "Encoder produced last_hidden_state sequence length: %d", token_count
                        )
                        # this is likely the correct mapping count
                        return max(1, token_count)
        except Exception as e:
            logger.debug("Encoder-run inference failed: %s", e, exc_info=True)

        # 4) Heuristic from pixel_values shape (cap to avoid huge appends)
        try:
            pv = inputs.get("pixel_values", None)
            if pv is not None:
                shape = tuple(pv.shape)
                if len(shape) == 4:
                    _, c, h, w = shape
                    # Estimated patch count using patch size ~16
                    patches_est = max(1, (h // 16) * (w // 16))
                    CAP = int(IMG_TOKENS_HEURISTIC_CAP)
                    val = min(patches_est, CAP)
                    logger.info(
                        """Heuristic estimated %d image tokens from pixel_values
                        shape %s (capped to %d)""",
                        patches_est,
                        shape,
                        val,
                    )
                    return int(val)
                elif len(shape) == 3:
                    seq_len = int(shape[1])
                    logger.info(
                        "Using pixel_values sequence length %d as image token count", seq_len
                    )
                    return max(1, seq_len)
        except Exception:
            logger.debug("Pixel-values heuristic failed", exc_info=True)

        # 5) Final fallback
        logger.warning("Unable to infer image token count; defaulting to 1")
        return 1

    # -------------------------
    # Public API
    # -------------------------
    def run_inference(
        self, image_path: Optional[str] = None, prompt: str = "Summarize the MRI scan"
    ) -> str:
        """
        Run multimodal generation. Returns generated text.
        If image_path is provided, ensures an appropriate number of image placeholder tokens
        are present in input_ids to align with image embeddings.
        """
        logger.info("Running MedGemma inference (image_path provided=%s)", bool(image_path))

        # Prepare inputs
        if image_path:
            image = Image.open(image_path).convert("RGB")
            # processor may return meta keys like 'num_crops'
            # — keep them for inspection but will filter later
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = self._move_tensors_to_device(inputs, self.model.device)

            tokenized = self.tokenizer(prompt, return_tensors="pt")
            tokenized = self._move_tensors_to_device(tokenized, self.model.device)

            # Keep pixel_values from processor; add text input_ids from tokenizer
            inputs["input_ids"] = tokenized.get("input_ids")
        else:
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            inputs = self._move_tensors_to_device(tokenized, self.model.device)

        # If image present, ensure text has image placeholder tokens aligned to image embeddings
        if "pixel_values" in inputs:
            image_token_str, image_token_id, _added = self._find_or_add_image_token()
            img_token_count = self._infer_image_token_count(inputs)
            logger.info(
                "Decided to use %d image placeholder tokens (token='%s' id=%s)",
                img_token_count,
                image_token_str,
                image_token_id,
            )

            # Safety cap to avoid insane sequence growth
            MAX_APPEND = int(IMG_TOKENS_MAX_APPEND)
            if img_token_count > MAX_APPEND:
                logger.warning(
                    "img_token_count %d exceeds MAX_APPEND %d; capping", img_token_count, MAX_APPEND
                )
                img_token_count = MAX_APPEND

            input_ids = inputs.get("input_ids")
            if input_ids is None:
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
                        # Append image tokens by default. If you need
                        # insertion at a marker, modify here.
                        inputs["input_ids"] = torch.cat([input_ids, img_tokens], dim=1)
                        logger.info(
                            "Appended %d image tokens to input_ids (new shape=%s).",
                            img_token_count,
                            inputs["input_ids"].shape,
                        )
                    else:
                        inputs["input_ids"] = torch.tensor(
                            [[image_token_id] * img_token_count], device=self.model.device
                        )
                        logger.info(
                            "Replaced non-tensor input_ids with %d image tokens.", img_token_count
                        )
                except Exception as e:
                    logger.warning(
                        """Failed to append image tokens (%s). Overwriting
                        input_ids with image tokens.""",
                        e,
                    )
                    inputs["input_ids"] = torch.tensor(
                        [[image_token_id] * img_token_count], device=self.model.device
                    )

        # Filter out unsupported keys to avoid generate() ValueError
        gen_inputs = self._filter_generate_inputs(inputs)

        # Generate
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**gen_inputs, max_new_tokens=512)
        except Exception as e:
            logger.error("Model.generate failed: %s", e, exc_info=True)
            raise

        # Decode
        try:
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            generated_text = [
                self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids
            ][0]

        logger.info("MedGemma inference completed.")
        return generated_text
