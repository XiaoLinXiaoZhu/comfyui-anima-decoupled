# Mechanism notes

Why the source/target split is possible, why the repetition form is effective, and what
the measurements do and do not establish.

---

## 1. The upstream contract

Anima's text path is fixed; the node fills two slots of it.

| Stage | Upstream location | Behaviour |
|---|---|---|
| Tokenization | `comfy/text_encoders/anima.py` → `AnimaTokenizer.tokenize_with_weights` | One text is tokenized twice: `qwen3_06b` (Qwen vocabulary, per-token weights forced to 1.0) and `t5xxl` (T5 vocabulary, per-token weights preserved). |
| Text encoding | same file → `AnimaTEModel.encode_token_weights` | Writes the `t5xxl_ids` and `t5xxl_weights` tensors into the conditioning entry's extra dict. |
| Model side | `comfy/model_base.py` → `Anima.extra_conds` | Reads `cross_attn`, `t5xxl_ids` and `t5xxl_weights` from each entry and calls `preprocess_text_embeds` itself. |
| Adapter | `comfy/ldm/anima/model.py` → `LLMAdapter` | Query = `Embedding(vocab, d)(t5xxl_ids)`; context = Qwen hidden states; 6 layers of self-attention with RoPE, cross-attention and MLP. The output is multiplied by `t5xxl_weights` per row and zero-padded to 512 rows; it is never truncated. |

`LLMAdapter(source_hidden_states, target_input_ids)` takes two inputs, places no
constraint between them, and produces one output. The pairing of a single text with
itself is a property of the upstream tokenizer wrapper, not a requirement of the model. The
adapter is also external to the DiT: it is completed before DiT training, and the DiT
consumes only the adapter's output, so replacing the source text requires no DiT change.

Since `extra_conds` performs the adapter forward pass, a custom node produces two items
only:

1. Qwen-side hidden states in `cross_attn` (already returned by
   `encode_from_tokens_scheduled`);
2. T5-side `t5xxl_ids` and `t5xxl_weights`.

No modification of upstream code or of the model is required.

## 2. What the DiT reads

**Rows are anchored to T5 token ids.** Each conditioning row's residual begins at
`Embedding(t5_token_id)`, and the row count equals the T5 token count of `t5_input`. The
row count is therefore determined by `t5_input`, not by `qwen_input`. Fixed prefixes,
delimiters and metadata placed on the target side consume row budget and change the
real-row to zero-row ratio.

**The DiT is order-blind.** Cross-attention in the DiT applies neither a mask nor RoPE;
RoPE is used only in self-attention. Conditioning rows therefore reach the image as an
unordered set, and binding information cannot be carried by ordering. It is carried by row
content, which is dominated by T5 token identity. Content to be drawn is written into
`t5_input` as complete phrases.

**Zero-padded rows participate.** Their keys are `RMSNorm(0) = 0`, giving an attention
logit of 0, and their values are 0; softmax nevertheless assigns probability mass to them.
A short and dense `t5_input` keeps the effective gain of the real rows high.

## 3. Why repetition is effective

Qwen3-0.6B is a causal model: in `A B C`, `A` cannot attend to `B` or `C`. Encoded as
`A1 B1 C1 A2 B2 C2`, the second copy can attend to the first, which gives later copies an
approximately bidirectional view. `PP【P】` retains a later copy as the conditioning rows,
so the row set has the size of a single pass while its content carries the earlier
context.

### Measurements

All values are `rel ‖Δ‖` (relative Frobenius difference) measured at the
**conditioning-vector level**, in fp32, on a single Anima checkpoint. They quantify how
far the conditioning moved, not whether images improved. Reference magnitudes from the
same setup: bf16 numerical noise ≈ 0.012; source rewritten into structured form ≈
0.22–0.42; unrelated long text ≈ 0.68–0.75.

| Finding | Result |
|---|---|
| Causal control | The first copy's hidden states are bit-identical between one pass and n passes (0.0000). |
| Backflow | Editing only the final tag displaces the shared-prefix tokens of the second copy by 0.118. The backflow carries content: two different edits produce directions with a cosine of 0.27, not 1. |
| Backflow dominates | Content backflow (0.55–0.58) exceeds generic preceding-text drift (0.19–0.34). |
| Direct form is diluted | `qwen_input = P P P` displaces the conditioning by 0.11–0.14: one third of the keys are the un-contextualized first copy, and a nearly orthogonal key-duplication effect is mixed in. |
| Prefix form is stronger | Retaining only the final copy displaces it by 0.37–0.51, i.e. **2.8–3.8×** the direct form, with a cleaner direction. |
| Saturation | The third copy reaches about 86–92% of the sixth copy's effect; the second copy already exceeds half. The gain decays quickly beyond three copies. |
| Not a volume control | Sensitivity to swapping the source content does not increase with the repetition count (×0.83–1.02): repetition changes what the source says, not how strongly it is weighted. |

### What is not established

These measurements were obtained from one checkpoint, a small prompt set and the
conditioning level. No calibrated relation between `rel ‖Δ‖` and image quality exists, and
the repetition form has not been shown to win a blind image comparison. It is a
mechanism-backed configuration to evaluate on the prompts at hand.

## 4. `strip_prefix` processing

```
source_text = qwen_prefix.strip() + " " + qwen_input.lstrip()
```

The Qwen encoder runs on `source_text` in one forward pass. With `strip_prefix=True` the
node then slices `cond[:, dropped:]`, where `dropped` is the measured length of the
longest common token prefix of `tokenize(source_text)` and `tokenize(qwen_prefix)`.

The value is measured rather than taken from `len(tokenize(prefix))`: whitespace merging at
the join point can change the token count by one, so comparison of the actual token-id
sequences is the only reliable way to determine how many rows belong to the prefix.

Because Qwen attention is causal, the prefix still influences every subsequent row; it does
not occupy rows of the adapter's input. The prefix therefore exerts influence without
becoming drawable content. If `dropped == 0`, the node raises an error instead of silently
retaining the prefix rows.
