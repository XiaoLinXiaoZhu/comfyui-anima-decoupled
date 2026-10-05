# Mechanism notes

Why the source/target split is possible, why the repetition recipe works, and what the
measurements do and do not establish.

---

## 1. The upstream contract

Anima's text path is fixed; this node only fills in two slots of it.

| Stage | Upstream location | Fact |
|---|---|---|
| Tokenization | `comfy/text_encoders/anima.py` → `AnimaTokenizer.tokenize_with_weights` | One text is tokenized **twice**: `qwen3_06b` (Qwen vocabulary, per-token weights forced to 1.0) and `t5xxl` (T5 vocabulary, per-token weights preserved). |
| CLIP encode | same file → `AnimaTEModel.encode_token_weights` | Puts `t5xxl_ids` and `t5xxl_weights` tensors into the conditioning entry's extra dict. |
| Model side | `comfy/model_base.py` → `Anima.extra_conds` | Reads `cross_attn`, `t5xxl_ids`, `t5xxl_weights` from each entry and calls `preprocess_text_embeds` itself. |
| Adapter | `comfy/ldm/anima/model.py` → `LLMAdapter` | Query = `Embedding(vocab, d)(t5xxl_ids)`; context = Qwen hidden states; 6 layers of (self-attention with RoPE + cross-attention + MLP). Output is multiplied by `t5xxl_weights` per row, then **zero-padded to 512 rows** (never truncated). |

Two independent inputs, one output. `LLMAdapter(source_hidden_states, target_input_ids)`
places no constraint between the two — the single-text pairing upstream produces is a
convention of the tokenizer wrapper, not a model requirement. The adapter also sits
outside the DiT: it is completed before DiT training and the DiT only consumes its
output, so replacing the source text needs no DiT change.

Because `extra_conds` runs the adapter, a custom node only has to produce:

1. Qwen-side hidden states in `cross_attn` (what `encode_from_tokens_scheduled` already
   returns),
2. T5-side `t5xxl_ids` / `t5xxl_weights`.

That is the entire trick — no patching, no model surgery.

## 2. What the DiT sees

**Rows are anchored to T5 token ids.** Every conditioning row's residual starts from
`Embedding(t5_token_id)`, and the row count equals the T5 token count of `t5_input`. This
is why `t5_input` — not `qwen_input` — decides how many rows the DiT sees. Fixed prefixes,
delimiters and metadata in the target side consume row budget and change the real-row :
zero-row ratio.

**The DiT is order-blind.** Cross-attention in the DiT applies no mask and no RoPE (RoPE
is used only in self-attention), so conditioning rows reach the image as an **unordered
set**. Binding information cannot ride on ordering; it has to be carried by row *content*,
which is dominated by the T5 token identity. Write what you want drawn into `t5_input`,
as complete phrases.

**Zero-padded rows still participate.** Their keys are `RMSNorm(0) = 0` (attention logit
0) and their values are 0, so softmax still spends probability mass on them. A short,
dense `t5_input` keeps the effective gain of the real rows high.

## 3. Why repetition works

Qwen3-0.6B is a causal model: in `A B C`, `A` cannot attend to `B` or `C`. Encoded as
`A1 B1 C1 A2 B2 C2`, the second copy can attend to the first, giving later copies a
near-bidirectional view. `PP【P】` keeps only a later copy as conditioning rows, so the
row set stays the size of a single pass while its content carries the earlier context.

### Measurements

All numbers are `rel ‖Δ‖` (relative Frobenius difference) at the **conditioning-vector
level**, fp32, on a single Anima checkpoint. They measure how much the conditioning
moved, not whether images improved. Reference magnitudes from the same setup: bf16
numerical noise ≈ 0.012; rewriting the source into structured form ≈ 0.22–0.42;
unrelated long text ≈ 0.68–0.75.

| Finding | Result |
|---|---|
| Causal control | The first copy's hidden states are bit-identical between one pass and n passes (0.0000). |
| Backflow | Editing only the last tag moves the shared-prefix tokens of the **second** copy by 0.118. It carries content: two different edits produce directions with cosine 0.27, not 1. |
| Backflow dominates | Content backflow (0.55–0.58) is larger than generic preceding-text drift (0.19–0.34). |
| Naive form is diluted | `qwen_input = P P P` moves the conditioning 0.11–0.14: one third of the keys are the un-contextualized first copy, and a nearly orthogonal key-duplication effect is mixed in. |
| `PP【P】` is stronger | Keeping only the final copy moves it 0.37–0.51 — **2.8–3.8×** the naive form — with a cleaner direction. |
| Saturation | Copy 3 reaches ~86–92% of copy 6's effect; copy 2 already exceeds half. Past three copies gains shrink fast. |
| Not a volume knob | Sensitivity to swapping the source content does not increase with repeats (×0.83–1.02): repetition changes *what* the source says, not *how loudly*. |

### What is not established

These measurements come from one checkpoint, a handful of prompts and the conditioning
level. There is no calibrated relation between `rel ‖Δ‖` and image quality, and the
recipe has not been shown to win a blind image comparison. Treat `PP【P】` as a
mechanism-backed experiment to run on your own prompts.

## 4. `strip_prefix` in detail

```
source_text = qwen_prefix.strip() + " " + qwen_input.lstrip()
```

The Qwen encoder runs on `source_text` in one forward pass. With `strip_prefix=True` the
node then slices `cond[:, dropped:]`, where `dropped` is the **measured** longest common
token prefix between `tokenize(source_text)` and `tokenize(qwen_prefix)`.

Measured, not `len(tokenize(prefix))`: whitespace merging at the join point can change the
token count by one, so comparing actual token-id sequences is the only reliable way to
know how many rows belong to the prefix.

Because Qwen attention is causal, the prefix still influences every subsequent row; it
just does not occupy rows of the adapter's input. That is the point: influence without
becoming drawable content. If `dropped == 0` the node raises instead of silently keeping
prefix rows.
