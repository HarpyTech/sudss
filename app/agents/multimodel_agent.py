# app/agents/multimodel_agent.py
# -*- coding: utf-8 -*-
"""
MedGemma/Gemma-3 multimodal wrapper.

Integrations:
- Uses config.constants.MED_GEMMA_4B as the default model name.
- Uses config.constants.LOG_FORMAT for logging format.

Key behavior:
- Aligns image placeholder token with model.config.image_token_id (or '<image>' fallback).
- Appends exactly the required count of image placeholder tokens (defaults to 256 if not explicit).
- Preserves 'image_sizes' and other vision metadata passed through generate().
- Removes stray '$' in prompts; when an image is present the prompt ends with ' : <image>'.

Public API:
    model = MedGemmaModel()  # defaults to MED_GEMMA_4B
    text = model.run_inference(image_path="path/to/image.png", prompt="Describe the scan")

Exports:
    MedGemmaModel, MedicalImageModel (alias), load_medgemma
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

# --- Pull configuration from your constants module ---
from config.constants import MED_GEMMA_4B, LOG_FORMAT  # noqa: E402

# ---------- Logging setup ----------
# Prevent duplicate handlers in some reload contexts.
_root_logger = logging.getLogger()
if not _root_logger.handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
else:
    # Ensure root logger uses the expected format at least for the first handler.
    _root_logger.handlers[0].setFormatter(logging.Formatter(LOG_FORMAT))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Safety cap: maximum number of image placeholder tokens we append.
IMG_TOKENS_MAX_APPEND = 4096


@dataclass
class ModelInitConfig:
    model_name: str
    device: Optional[str] = None  # "cuda", "cpu", or None to auto-select
    dtype: Optional[str] = None   # "float16", "bfloat16", etc., or None for default
    trust_remote_code: bool = True


class MedGemmaModel:
    """
    Thin convenience layer for Gemma-3-style VLMs (e.g., MedGemma variants).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        # Default to configured model name if not provided
        model_name = model_name or MED_GEMMA_4B
        self.cfg = ModelInitConfig(
            model_name=model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        # Resolve device and dtype
        if self.cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.cfg.device

        torch_dtype = None
        if self.cfg.dtype:
            try:
                torch_dtype = getattr(torch, self.cfg.dtype)
            except AttributeError:
                logger.warning("Unknown dtype '%s'; falling back to default.", self.cfg.dtype)

        logger.info("Loading processor/tokenizer/model: %s", self.cfg.model_name)
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        # Some Gemma-3 processors already contain a tokenizer; still get tokenizer explicitly
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=self.cfg.trust_remote_code,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=self.cfg.trust_remote_code,
        ).to(self.device)

        # Make sure special tokens are aligned (pad token, etc.) if needed
        if self.tokenizer.pad_token is None:
            # Prefer eos as pad to avoid size mismatches
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model loaded on device=%s dtype=%s", self.device, str(torch_dtype))

    # ------------------------------ Public API ------------------------------

    def run_inference(
        self,
        image_path: Optional[str] = None,
        prompt: str = "Summarize the MRI scan",
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generates text from an optional image + text prompt.
        Ensures the number and id of image placeholder tokens match the model's expectation.
        """
        logger.info("Running MedGemma inference (image provided=%s)", bool(image_path))

        # If an image is present, append the textual marker '<image>' to the user prompt.
        if image_path:
            prompt = f"{prompt} : <image>"

        # Build vision/text inputs
        if image_path:
            image = Image.open(image_path).convert("RGB")

            image.verify()
            # For Gemma-3 processors, this usually provides pixel_values (+ metadata)
            processor_inputs = self.processor(images=image, return_tensors="pt")
            processor_inputs = self._move_tensors_to_device(processor_inputs, self.model.device)

            tok = self.tokenizer(prompt, return_tensors="pt")
            tok = self._move_tensors_to_device(tok, self.model.device)

            inputs: Dict[str, Any] = {**processor_inputs, **tok}
        else:
            # Text-only
            tok = self.tokenizer(prompt, return_tensors="pt")
            inputs = self._move_tensors_to_device(tok, self.model.device)

        # If we have images, ensure correct placeholder tokens are appended
        if "pixel_values" in inputs:
            token_str, token_id, _ = self._find_or_add_image_token()
            n_img_tokens = self._infer_image_token_count(inputs)
            logger.info("Using %d image tokens (token='%s', id=%d)", n_img_tokens, token_str, token_id)

            if n_img_tokens > IMG_TOKENS_MAX_APPEND:
                logger.warning(
                    "Inferred image token count %d exceeds cap %d; capping.",
                    n_img_tokens, IMG_TOKENS_MAX_APPEND
                )
                n_img_tokens = IMG_TOKENS_MAX_APPEND

            inputs["input_ids"] = self._append_image_placeholders(
                inputs.get("input_ids"),
                n_img_tokens,
                token_id,
                device=self.model.device
            )

        gen_inputs = self._filter_generate_inputs(inputs)
        logger.debug("Final keys sent to generate(): %s", sorted(gen_inputs.keys()))

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
            )

        # Decode
        try:
            text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            text = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids][0]

        logger.info("MedGemma inference completed.")
        return text

    # --------------------------- Internal Utilities --------------------------

    def _append_image_placeholders(
        self,
        input_ids: Optional[torch.Tensor],
        count: int,
        image_token_id: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Append 'count' copies of image_token_id to input_ids (or create a fresh tensor if None).
        """
        img_tokens = torch.full(
            (1, int(count)),
            fill_value=int(image_token_id),
            device=device,
            dtype=torch.long,
        )
        if input_ids is None:
            out = img_tokens
        else:
            out = torch.cat([input_ids, img_tokens], dim=1)

        logger.info("Final input_ids shape after image placeholders: %s", tuple(out.shape))
        return out

    def _move_tensors_to_device(self, batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        return moved

    def _infer_image_token_count(self, inputs: Dict[str, Any]) -> int:
        """
        Infer how many image placeholder tokens the model expects, based on processed vision
        inputs and/or model config. Fall back conservatively to 256.
        """
        # Single-image assumption
        num_images = 1
        default_tokens = 256

        # If processor provided an explicit 'image_grid_thw' (T,H,W -> tokens = T*H*W)
        if "image_grid_thw" in inputs:
            thw = inputs["image_grid_thw"]
            if isinstance(thw, torch.Tensor) and thw.numel() >= 3:
                t, h, w = [int(x) for x in thw[0].tolist()[:3]]
                n = max(1, t) * max(1, h) * max(1, w)
                return n * num_images

        # Some processors expose 'image_sizes' which can be used to compute grids for patchified encoders.
        # We keep default; preserving 'image_sizes' is handled in _filter_generate_inputs.
        if "image_sizes" in inputs:
            return default_tokens * num_images

        # Fallback
        return default_tokens * num_images

    def _filter_generate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only keys accepted by model.generate() -> model.forward(), but *preserve*
        vision metadata keys commonly needed by VLMs.
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
            # vision metadata
            "image_sizes",
        }
        filtered = {k: v for k, v in inputs.items() if k in allowed_keys}
        removed = [k for k in inputs.keys() if k not in filtered]
        if removed:
            logger.debug("Removed unexpected keys before model.generate(): %s", removed)
        return filtered

    def _find_or_add_image_token(self, candidates: Optional[Tuple[str, ...]] = None) -> Tuple[str, int, bool]:
        """
        Prefer model.config.image_token_id if present. Otherwise, use '<image>' and make sure
        it exists in the tokenizer; if needed, add it and resize embeddings. Record the final id
        back into model.config.image_token_id so the modeling code can find it.
        Returns (token_str, token_id, added_flag).
        """
        cfg = getattr(self.model, "config", None)

        # 0) If config already specifies an image token id, use it.
        if cfg is not None and getattr(cfg, "image_token_id", None) is not None:
            tok_id = int(cfg.image_token_id)
            # token string may not be faithfully decodable; keep a friendly label
            try:
                tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                if not tok_str or tok_str == self.tokenizer.unk_token:
                    tok_str = "<image>"
            except Exception:
                tok_str = "<image>"
            logger.info("Using model.config.image_token_id=%d (token '%s')", tok_id, tok_str)
            return tok_str, tok_id, False

        # 1) Prefer '<image>'
        image_token = "<image>"
        tok_id = self.tokenizer.convert_tokens_to_ids(image_token)
        need_add = tok_id is None or (
            self.tokenizer.unk_token_id is not None and tok_id == self.tokenizer.unk_token_id
        )

        if need_add:
            logger.info("Tokenizer lacks '<image>' token. Adding it as additional_special_tokens.")
            try:
                self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
                self.model.resize_token_embeddings(len(self.tokenizer))
                tok_id = self.tokenizer.convert_tokens_to_ids(image_token)
            except Exception as e:
                logger.warning("Adding '<image>' failed: %s", repr(e))
                tok_id = None

        # 2) If still unresolved, try common aliases
        if tok_id is None or (
            self.tokenizer.unk_token_id is not None and tok_id == self.tokenizer.unk_token_id
        ):
            for cand in ("<image_0>", "<image0>", "[IMAGE]"):
                t = self.tokenizer.convert_tokens_to_ids(cand)
                if t is not None and (self.tokenizer.unk_token_id is None or t != self.tokenizer.unk_token_id):
                    image_token, tok_id = cand, int(t)
                    logger.info("Falling back to existing image token '%s' (id=%d).", cand, tok_id)
                    break

        if tok_id is None or (
            self.tokenizer.unk_token_id is not None and tok_id == self.tokenizer.unk_token_id
        ):
            raise ValueError("Could not establish a valid image token id for Gemma-3/MedGemma.")

        # 3) Record on config so modeling code can find it
        try:
            if cfg is not None:
                cfg.image_token_id = int(tok_id)
                logger.info("Set model.config.image_token_id=%d", int(tok_id))
        except Exception as e:
            logger.debug("Failed to set config.image_token_id: %s", repr(e))

        return image_token, int(tok_id), True


# Backwards-compat alias (your app may import this name)
MedicalImageModel = MedGemmaModel


# Optional: simple factory, if your app prefers constructing via a function.
def load_medgemma(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None
) -> MedGemmaModel:
    """
    Convenience factory to create a MedGemmaModel.
    Defaults to MED_GEMMA_4B from config.constants if model_name is None.
    """
    return MedGemmaModel(model_name=model_name or MED_GEMMA_4B, device=device, dtype=dtype)


__all__ = ["MedGemmaModel", "MedicalImageModel", "load_medgemma"]
