"""Decoupled conditioning construction for Anima: Qwen side vs. T5 side.

This module is pure mechanism. It does not parse XML, does not clean tags and
does not validate formats: the caller decides the prefix, what the Qwen side
receives and what the T5 side receives.

Contract (verified against upstream ComfyUI, ``comfy/model_base.py`` ->
``Anima.extra_conds`` and ``comfy/ldm/anima/model.py``):

- ``Anima.extra_conds`` reads ``cross_attn``, ``t5xxl_ids`` and ``t5xxl_weights``
  from every conditioning entry and then calls ``preprocess_text_embeds`` itself
  (LLM adapter + per-row weights + zero padding up to 512 rows).
- Therefore this module only has to produce: Qwen-side hidden states plus
  T5-side token ids and weights.

Behaviour:

- The Qwen side encodes ``qwen_prefix + " " + qwen_input`` (just ``qwen_input``
  when the prefix is empty).
- With ``strip_prefix=True`` (the default) the rows belonging to the prefix are
  cut away, using the longest common token prefix of the joined text. The prefix
  still reaches the following rows through causal attention, but does not enter
  the adapter as a conditioning row of its own.
- ``t5xxl_ids`` / ``t5xxl_weights`` are taken from ``t5_input``, so the number of
  conditioning rows and their token anchors are decided by the T5 side.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = [
    "common_prefix_len",
    "qwen_ids",
    "t5_ids_and_weights",
    "join_prefix",
    "build_conditioning",
]


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token-id sequences.

    This must be measured, it cannot be derived from ``len(tokenize(prefix))``:
    whitespace merging at the join point between prefix and body can change the
    token count by one.
    """
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def qwen_ids(tokens: dict) -> list[int]:
    """Token ids of the Qwen3-0.6B (source side) tokenization."""
    try:
        pairs = tokens["qwen3_06b"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "clip did not tokenize with Anima's dual tokenizer (no 'qwen3_06b' "
            "entry); connect the CLIP output of an Anima checkpoint"
        ) from exc
    return [pair[0] for pair in pairs]


def t5_ids_and_weights(tokens: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and per-token weights of the T5 (target side) tokenization."""
    try:
        pairs = tokens["t5xxl"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "clip did not tokenize with Anima's dual tokenizer (no 't5xxl' "
            "entry); connect the CLIP output of an Anima checkpoint"
        ) from exc
    ids = torch.tensor([pair[0] for pair in pairs], dtype=torch.int)
    weights = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32)
    return ids, weights


def join_prefix(qwen_prefix: str, qwen_input: str) -> str:
    """Join convention: exactly one space between prefix and body.

    An empty (or whitespace-only) prefix returns the body unchanged.
    """
    prefix = qwen_prefix.strip()
    if not prefix:
        return qwen_input
    return f"{prefix} {qwen_input.lstrip()}"


def build_conditioning(
    clip,
    qwen_input: str,
    t5_input: str,
    qwen_prefix: str = "",
    strip_prefix: bool = True,
) -> tuple[list, int, int]:
    """Build Anima conditioning from two independent texts.

    Args:
        clip: Anima CLIP object (``AnimaTokenizer`` based).
        qwen_input: body of the text handed to the Qwen3-0.6B source encoder.
        t5_input: text handed to the T5 target side; it decides the number of
            conditioning rows and the per-row token anchors.
        qwen_prefix: optional text prepended to ``qwen_input`` on the Qwen side.
        strip_prefix: when true, remove the rows of ``qwen_prefix`` from the
            source hidden states before they reach the adapter.

    Returns:
        ``(conditioning, dropped_qwen_rows, t5_rows)``.
    """
    source_text = join_prefix(qwen_prefix, qwen_input)
    source_tokens = clip.tokenize(source_text)

    dropped = 0
    if strip_prefix and qwen_prefix.strip():
        dropped = common_prefix_len(
            qwen_ids(source_tokens), qwen_ids(clip.tokenize(qwen_prefix.strip()))
        )
        if dropped == 0:
            raise ValueError(
                "qwen_prefix shares no removable token prefix with the joined "
                "text; check the prefix, or turn strip_prefix off"
            )

    target_ids, target_weights = t5_ids_and_weights(clip.tokenize(t5_input))

    conditioning = []
    for cond, extra in clip.encode_from_tokens_scheduled(source_tokens):
        new_extra = dict(extra)
        new_extra["t5xxl_ids"] = target_ids
        new_extra["t5xxl_weights"] = target_weights
        conditioning.append([cond[:, dropped:] if dropped else cond, new_extra])

    return conditioning, dropped, int(target_ids.shape[0])
