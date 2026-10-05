# Mechanism notes

What the node actually hands to Anima, why decoupling works at all, and what is
(and is not) established about the repetition recipe.

---

## 1. Upstream contract

Anima's text path is fixed; this node only fills in two slots of it.

| Stage | Upstream location | Fact |
|---|---|---|
| Tokenization | `comfy/text_encoders/anima.py` → `AnimaTokenizer.tokenize_with_weights` | One text is tokenized **twice**: `qwen3_06b` (Qwen2 vocab, per-token weights forced to 1.0) and `t5xxl` (T5 vocab, per-token weights preserved). |
| CLIP encode | same file → `AnimaTEModel.encode_token_weights` | Puts `t5xxl_ids` and `t5xxl_weights` tensors into the conditioning entry's extra dict. |
| Model side | `comfy/model_base.py` → `Anima.extra_conds` | Reads `cross_attn`, `t5xxl_ids`, `t5xxl_weights` from each entry and calls `preprocess_text_embeds` itself. |
| Adapter | `comfy/ldm/anima/model.py` → `LLMAdapter` | Query = `Embedding(vocab, d)(t5xxl_ids)`; context = Qwen hidden states; 6 layers of (self-attn with RoPE + cross-attn + MLP). Output is multiplied by `t5xxl_weights` per row, then **zero-padded to 512 rows** (never truncated). |

Because `extra_conds` does the adapter, a custom node needs to produce exactly
two things:

1. Qwen-side hidden states in `cross_attn` (what `encode_from_tokens_scheduled`
   already returns),
2. T5-side `t5xxl_ids` / `t5xxl_weights`.

That is the entire trick — no patching, no model surgery.

## 2. Consequences of the contract

**Rows are anchored to T5 token ids.** Every conditioning row's residual starts
from `Embedding(t5_token_id)`, and the row count equals the T5 token count of
`t5_input`. This is why `t5_input` — not `qwen_input` — decides how many rows the
DiT sees. Putting fixed prefixes, delimiters or metadata in the target side
consumes row budget and changes the real-row : zero-row ratio.

**The DiT is order-blind.** Anima's DiT applies cross-attention with no mask and
no RoPE (RoPE is used only in self-attention), so the conditioning rows reach the
image as an **unordered set**. Binding information therefore cannot ride on
ordering; it has to be carried by row *content*, which is dominated by the T5
token identity. Practical read: write what you want drawn into `t5_input`, as
complete phrases.

**Zero-padded rows still participate.** The padding rows are not masked out:
their keys are `RMSNorm(0) = 0` (attention logit 0) and their values are 0, so
softmax still spends probability mass on them. Short, dense `t5_input` keeps the
effective gain of the real rows high.

## 3. `strip_prefix`

```
source_text = qwen_prefix.strip() + " " + qwen_input.lstrip()
```

The Qwen encoder runs on `source_text` in one forward pass. If
`strip_prefix=True`, the node then slices `cond[:, dropped:]`, where `dropped` is
the **measured** longest common token prefix between `tokenize(source_text)` and
`tokenize(qwen_prefix)`.

Why measured and not `len(tokenize(prefix))`: whitespace merging at the join
point can change the token count by one. Comparing actual token-id sequences is
the only reliable way to know how many rows belong to the prefix.

Because Qwen attention is causal, the prefix still influences every subsequent
row — it just does not occupy rows of its own in the adapter input. That is the
whole point: influence without becoming drawable content.

If `dropped == 0` the node raises rather than silently keeping prefix rows.

## 4. Repetition, `PP【P】`

### Why it can work

Qwen3-0.6B is a causal model: in `A B C`, token `A` cannot attend to `B` or `C`.
Encoding `A1 B1 C1 A2 B2 C2` lets the second copy attend to the first, giving the
later copies a near-bidirectional view of the text. `PP【P】` keeps only that
later copy as conditioning rows, so the row set stays the size of a single pass.

### Measurements

All numbers below are `rel ‖Δ‖` (relative Frobenius difference) at the
**conditioning-vector level**, fp32, on a single Anima checkpoint. They measure
*how much the conditioning moved*, not whether images improved. Reference
magnitudes from the same setup: bf16 numerical noise ≈ 0.012; rewriting the
source into structured form ≈ 0.22–0.42; unrelated long text ≈ 0.68–0.75.

| Finding | Result |
|---|---|
| Causal control | First copy's hidden states are bit-identical between one pass and n passes (0.0000). |
| Backflow (K1) | Editing only the **last** tag moves the shared prefix tokens in the **second** copy by 0.118 — later content demonstrably reaches earlier positions, and it carries content (the direction for two different edits has cosine 0.27, not 1). |
| Backflow dominates (K2) | Content backflow (0.55–0.58) is larger than generic preceding-text drift (0.19–0.34). |
| Naive form is diluted (K3) | `qwen_input = P P P` moves the conditioning only 0.11–0.14, because one third of the keys are the un-contextualized first copy, and it mixes in a nearly orthogonal key-duplication effect. |
| `PP【P】` is stronger (K3) | Keeping only the final copy moves the conditioning 0.37–0.51 — **2.8–3.8×** the naive form — with a cleaner direction. |
| Saturation (K4) | Copy 3 reaches ~86–92% of copy 6's effect; copy 2 already exceeds half. Past three copies gains shrink fast. |
| Not a volume knob (K5) | Sensitivity to swapping the source content does not increase with repeats (×0.83–1.02), so repetition changes *what* the source says, not *how loudly*. |

### What is not established

These measurements were made on one checkpoint at the conditioning level with a
handful of prompts. There is no calibrated relation between `rel ‖Δ‖` and image
quality, and the repetition recipe has not been shown to beat a single pass in
blind image comparison. Treat `PP【P】` as a mechanism-backed experiment to run on
your own prompts, not as a guaranteed improvement.

## 5. Practical rules

1. `t5_input` is what gets drawn — keep it to content, phrased as complete units.
2. `qwen_input` contextualizes; it is the right home for prose, structure and
   metadata.
3. `qwen_prefix` + `strip_prefix` = influence without rows. Use it for house
   style and for the repetition recipe.
4. Row count equals `t5_input`'s T5 token count. Keep it short and dense.
5. One variable at a time when comparing; per-seed variation can swamp the effect.
