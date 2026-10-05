# Usage guide

Recipes first, then wiring, comparison method and troubleshooting. Node reference and
mechanism are in the [README](../README.md) and [mechanism notes](mechanism.md).

---

## 1. Recipes

### 1.1 Repeat the prompt — `PP【P】`

`P` is your prompt. `PP【P】` means: encode it three times, keep only the third copy.

```
qwen_prefix  = P P
qwen_input   = P
t5_input     = <the prompt you normally use>
strip_prefix = true
```

Paste the prompt twice into `qwen_prefix` and once into `qwen_input`. The three copies
must be identical text. Row count is unchanged from a single pass; the first two copies
are removed by token-level longest common prefix. Start at three copies — two already
gets most of the effect, past three the gains shrink quickly.

**Repeat only the body.** If your prompt has a boilerplate head (quality tags, artist
tags) that you do not want in the Qwen context, repeat just the body:

```
qwen_prefix  = <body> <body>
qwen_input   = <body>
t5_input     = <quality head>, <body tags>, <artist tags>
strip_prefix = true
```

The head still reaches the image through `t5_input`; it just no longer contextualizes
the Qwen side.

**Reference variant for comparison only.** Putting the repetitions directly into
`qwen_input` leaves every copy visible to the adapter:

```
qwen_input   = P P P
```

At the conditioning level this moves about a third as far as `PP【P】` and mixes in a
key-duplication effect. Expect it to score lower, but judge on images.

### 1.2 Tags on T5, narrative on Qwen

```
qwen_input = A courier leans on a rusted railing above a flooded street; neon signs
             behind her reflect in the puddles. Rain streaks across the lens and the
             background falls out of focus behind her left shoulder.
t5_input   = 1girl, solo, courier jacket, rain, puddle, neon sign, railing, night,
             from side, shallow depth of field, (backlight:1.2)
```

`t5_input` is what gets drawn; keep it to content, phrased as complete units. Everything
that is not drawable — prose, structure, comments — belongs on the Qwen side, where it
costs no conditioning rows.

Two things worth knowing before you lean on the Qwen side:

- Meta-text placed in `t5_input` was observed to render as visible text in the image.
- The Qwen channel has less leverage than the T5 side (roughly the 9% level for a
  pure-tag target in our measurements). Use it to disambiguate and steer; do not expect
  it to override what the tags say.

## 2. Wiring

```
Load Checkpoint ─┬─ CLIP ──> Anima Decoupled Conditioning ── CONDITIONING ──> KSampler (positive)
                 │                    ▲
                 │        qwen_input, t5_input, qwen_prefix
                 └─ MODEL ──────────────────────────────────────────────────> KSampler
```

1. Delete or disconnect the positive `CLIPTextEncode`.
2. Add **Anima Decoupled Conditioning** (category `conditioning`).
3. Connect `CLIP` → `clip`, and `conditioning` → the sampler's positive input.
4. Leave the negative path as a normal `CLIPTextEncode` if you use CFG > 1. Most Anima
   turbo setups run at CFG 1, where the negative input is inert; keeping it costs nothing
   and makes a later CFG change work.

The node must receive Anima's dual tokenizer through `clip`. Any other model's CLIP is
rejected with an error naming Anima.

## 3. Comparing variants

- Change one input at a time; keep seed, sampler, steps, CFG, resolution and model fixed.
- Use the same seed across variants and look at them side by side.
- Disable prompt randomization / wildcard nodes while comparing.
- Once you have a verdict, change the seed and check that it survives — per-seed variation
  can be larger than the effect you are chasing.
- Record the exact widget values next to each image; `PP【P】` output is easy to
  misattribute to the wrong string.

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `clip did not tokenize with Anima's dual tokenizer` | The `CLIP` input is not an Anima checkpoint. Load the Anima model and use its CLIP output. |
| `qwen_prefix shares no removable token prefix` | `strip_prefix` is on but the prefix is not a token-level prefix of the joined text. Check for leading junk, or turn `strip_prefix` off. |
| Conditioning looks identical to a plain encode | Expected when `qwen_input == t5_input` and `qwen_prefix` is empty: the node then reproduces the upstream single-text path. |
| Model output shows your prompt as text | The repeated text landed in `t5_input`. Keep repetitions on the Qwen side. |

## 5. FAQ

**Does it need a patched ComfyUI?**
No. It returns `cross_attn` plus `t5xxl_ids` / `t5xxl_weights` in the conditioning entry;
upstream `Anima.extra_conds` does the rest.

**Why 512 rows?**
That is upstream's padding target for the adapter output. This node does not pad;
`Anima.extra_conds` does.

**Can I use it for a negative prompt?**
It produces one conditioning. Add a second instance if your CFG makes negatives
meaningful.

**Can I use it with a non-Anima model?**
No — the mechanism is Anima's dual-tokenizer contract.
