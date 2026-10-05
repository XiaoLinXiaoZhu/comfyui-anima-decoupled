"""ComfyUI node wrapper around :mod:`anima_decoupled`."""

from __future__ import annotations

import logging

from .anima_decoupled import build_conditioning


class AnimaDecoupledConditioning:
    """Anima decoupled conditioning: Qwen and T5 receive different texts.

    Mechanism only, no format knowledge: no XML parsing, no tag cleaning, no
    format validation.

    - ``qwen_prefix``: optional text prepended to ``qwen_input`` (one space in
      between). It shapes the Qwen hidden states only.
    - ``qwen_input``: body of the text handed to the Qwen (source) encoder.
    - ``t5_input``: text handed to the T5 (target) side. It decides the number
      of conditioning rows and their token anchors.
    - ``strip_prefix``: on by default. When on, the rows that belong to
      ``qwen_prefix`` are cut from the source hidden states: the prefix still
      reaches the following rows through causal attention but does not become a
      conditioning row of its own.
    """

    SEARCH_ALIASES = [
        "anima decoupled",
        "anima conditioning",
        "prefix xml conditioning",
        "repeat prompt",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "qwen_input": ("STRING", {"multiline": True, "default": ""}),
                "t5_input": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "qwen_prefix": ("STRING", {"multiline": True, "default": ""}),
                "strip_prefix": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Anima only: feed independent texts to the Qwen (source) and T5 "
        "(target) sides; an optional prefix shapes the Qwen hidden states "
        "without becoming a conditioning row."
    )

    def encode(
        self,
        clip,
        qwen_input: str,
        t5_input: str,
        qwen_prefix: str = "",
        strip_prefix: bool = True,
    ):
        conditioning, dropped, target_rows = build_conditioning(
            clip, qwen_input, t5_input, qwen_prefix, strip_prefix
        )
        logging.info(
            "AnimaDecoupledConditioning: t5_rows=%d, stripped_qwen_rows=%d",
            target_rows,
            dropped,
        )
        return (conditioning,)
