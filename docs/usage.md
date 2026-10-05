# Usage guide

This guide covers wiring the node into a workflow, the recipes worth copying,
how to A/B them honestly, and what to do when something looks wrong.

---

## 1. Prerequisites

- A ComfyUI build with Anima support (upstream `>= 0.11.0`; developed on `0.37.0`).
- An Anima checkpoint loaded through `Load Checkpoint` (or `UNETLoader` + `CLIPLoader`
  for a split-file setup) — the `CLIP` output must be Anima's dual tokenizer
  (Qwen3-0.6B + T5). Any other model's `CLIP` is rejected with a clear error.

## 2. Wiring

The node is a drop-in replacement for the **positive** `CLIPTextEncode`:

```
Load Checkpoint ─┬─ CLIP ──> Anima Decoupled Conditioning ── CONDITIONING ──> KSampler (positive)
                 │                    ▲
                 │        qwen_input, t5_input, qwen_prefix
                 └─ MODEL ──────────────────────────────────────────────────> KSampler
```

Steps:

1. Delete (or disconnect) the positive `CLIPTextEncode`.
2. Add **Anima Decoupled Conditioning** (`conditioning` category).
3. Connect `CLIP` → `clip`.
4. Connect `conditioning` → the sampler's positive input.
5. Leave the **negative** path as a normal `CLIPTextEncode` if you use CFG > 1.
   (Anima turbo-style sampling at CFG 1 makes the negative input inert; keep it
   anyway so switching CFG later still works.)

There is no second text channel in the model: both sides end up in the same
512-row conditioning. The node lets you decide **what each side contributes**,
not whether the contributions exist.

## 3. Cookbook

### 3.1 Narrative for Qwen, tags for T5

```
qwen_input   = A courier rests against a rusted railing; neon puddles reflect
               the sign above her. Rain streaks the lens, shallow focus.
t5_input     = 1girl, solo, courier jacket, rain, neon, puddle, railing, night
qwen_prefix  =
strip_prefix = true
```

Use this when you want prose-level composition control without letting prose
characters leak into the conditioning rows.

### 3.2 Repeat the prompt — `PP【P】`

`P` is your prompt. `PP【P】` = encode it three times, keep only the third copy.

```
qwen_prefix  = P P
qwen_input   = P
t5_input     = <the tag string you normally use>
strip_prefix = true
```

Notes:

- The three copies must be **identical text**. Paste the prompt into
  `qwen_prefix`, paste it again so it appears twice, and once more into
  `qwen_input`.
- Row count is unchanged from a single pass: the first two copies are removed by
  token-level longest common prefix.
- Start at three copies. Two already gets most of the effect; past three the
  gains shrink quickly.

#### Repeating only the body

If your prompt has a boilerplate head (quality tags, artist tags) that you want
to keep out of the Qwen context entirely, repeat only the body:

```
qwen_prefix  = <body> <body>
qwen_input   = <body>
t5_input     = <quality head>, <body tags>, <artist tags>
strip_prefix = true
```

The head still reaches the image through `t5_input`; it just no longer
contextualizes the Qwen side.

#### Reference variant: all copies visible

For A/B comparison only. Putting the repetitions directly into `qwen_input`
leaves every copy visible to the adapter:

```
qwen_input   = P P P
```

At the conditioning level this moves roughly a third as far as the `PP【P】`
form and mixes in a key-duplication effect. Expect it to be the weaker of the
two, but judge on images.

### 3.3 A fixed prefix that never becomes a row

```
qwen_prefix  = cinematic still, 35mm, controlled contrast
qwen_input   = <your description>
t5_input     = <your tags>
strip_prefix = true
```

Good for house style and framing hints. Turning `strip_prefix` off keeps the
prefix as real conditioning rows — which is usually **not** what you want for
meta-text: rows in the conditioning are what the DiT reads, and meta-text there
has been observed to render as visible text in the image.

### 3.4 Row budget

Live conditioning rows = T5 token count of `t5_input`; upstream zero-pads the
rest up to 512. Long structural addenda in `t5_input` consume rows and shift the
real-row : zero-row ratio. Move anything that is not drawable to the Qwen side.

## 4. A/B testing honestly

- Vary **one** input at a time; keep seed, sampler, steps, CFG, resolution and
  model fixed.
- Use the same seed across variants and compare side by side.
- Disable prompt randomization / wildcard nodes while comparing.
- Change the seed **after** you have a verdict, to check the verdict survives —
  per-seed differences can be larger than the effect you are chasing.
- Record the exact widget values next to each image; `PP【P】` output is easy to
  misattribute to the wrong string.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `clip did not tokenize with Anima's dual tokenizer` | The `CLIP` input is not an Anima checkpoint. Load the Anima model and use **its** CLIP output. |
| `qwen_prefix shares no removable token prefix` | `strip_prefix` is on but the prefix is not a token-level prefix of the joined text. Check for leading junk, or set `strip_prefix` off. |
| Conditioning looks identical to a plain encode | Expected when `qwen_input == t5_input` and `qwen_prefix` is empty — the node then reproduces the upstream single-text path. |
| Nothing changes when editing `t5_input` | `t5_input` drives the rows; but if the sampler's positive is not connected to this node, the edit goes nowhere. Re-check the wiring. |
| Repeating the prompt makes output text appear in the image | The repeated text landed in `t5_input`. Keep repetitions on the Qwen side (`qwen_prefix`), not in the target side. |

## 6. FAQ

**Does it need a patched ComfyUI?**
No. It only returns `cross_attn` plus `t5xxl_ids` / `t5xxl_weights` in the
conditioning entry; upstream `Anima.extra_conds` does the rest.

**Can I use it with a non-Anima model?**
No — the whole mechanism is Anima's dual-tokenizer contract.

**Why 512 rows?**
That is upstream's padding target for the adapter output. This node does not pad;
`Anima.extra_conds` does.

**Does it support negative prompts on the Qwen side?**
The node produces one conditioning. Use a second instance for a negative encode
if your CFG makes negatives meaningful.
