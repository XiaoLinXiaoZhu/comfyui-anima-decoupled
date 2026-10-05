# Usage guide

Configurations first, then wiring, comparison method and troubleshooting. The node
reference and the mechanism are in the [README](../README.md) and
[mechanism notes](mechanism.md).

---

## 1. Configurations

### 1.1 Prompt repetition — `PP【P】`

`P` denotes the prompt. `PP【P】` means: encode it three times and keep only the third copy
as conditioning rows.

```
qwen_prefix  = P P
qwen_input   = P
t5_input     = <the prompt normally used>
strip_prefix = true
```

The prompt is entered twice into `qwen_prefix` and once into `qwen_input`; the three
copies must be identical. The row count is unchanged from a single pass, because the
first two copies are removed by token-level longest common prefix. Three copies is the
recommended starting point: two copies already provide most of the effect, and the gain
decays quickly beyond three.

**Repeating the body only.** When the prompt carries a boilerplate head (quality tags,
artist tags) that must not enter the Qwen context, only the body is repeated:

```
qwen_prefix  = <body> <body>
qwen_input   = <body>
t5_input     = <quality head>, <body tags>, <artist tags>
strip_prefix = true
```

The head still reaches the image through `t5_input`; it no longer contextualizes the Qwen
side.

**Reference form for comparison only.** Writing the repetitions directly into
`qwen_input` leaves every copy visible to the adapter:

```
qwen_input   = P P P
```

At the conditioning level this form displaces about one third as far as `PP【P】` and
mixes in a key-duplication effect. It is included for A/B comparison, not as a
recommended configuration.

### 1.2 Tags on T5, narrative on Qwen

```
qwen_input = A courier leans on a rusted railing above a flooded street; neon signs
             behind her reflect in the puddles. Rain streaks across the lens and the
             background falls out of focus behind her left shoulder.
t5_input   = 1girl, solo, courier jacket, rain, puddle, neon sign, railing, night,
             from side, shallow depth of field, (backlight:1.2)
```

`t5_input` determines what is drawn; content there is expressed as complete phrases.
Everything that is not drawable — prose, structure, comments — is placed on the Qwen
side, where it consumes no conditioning rows.

Two properties of the Qwen channel constrain its use:

- Meta-text placed in `t5_input` was observed to be rendered as visible text in the image.
- The Qwen channel has less influence on the final conditioning than the T5 side (roughly
  the 9% level for a pure-tag target in measurement). It is applicable to disambiguation
  and steering; it cannot be expected to override the content of `t5_input`.

## 2. Wiring

```
Load Checkpoint ─┬─ CLIP ──> Anima Decoupled Conditioning ── CONDITIONING ──> KSampler (positive)
                 │                    ▲
                 │        qwen_input, t5_input, qwen_prefix
                 └─ MODEL ──────────────────────────────────────────────────> KSampler
```

1. Disconnect or delete the positive text-encoding node.
2. Add **Anima Decoupled Conditioning** (category `conditioning`).
3. Connect `CLIP` to `clip`, and `conditioning` to the sampler's positive conditioning
   input.
4. Leave the negative path as a conventional text-encoding node when CFG > 1. Anima turbo
   configurations generally run at CFG 1, where the negative input has no effect; retaining
   it costs nothing and keeps a later CFG change functional.

The `clip` input must receive Anima's dual tokenizer. Any other model's CLIP output is
rejected with an error that names Anima.

## 3. Comparing variants

- Change one input at a time; hold seed, sampler, steps, CFG, resolution and model fixed.
- Use the same seed across variants and inspect them side by side.
- Disable prompt randomization and wildcard nodes for the duration of the comparison.
- After reaching a verdict, change the seed and confirm the verdict holds; per-seed
  variation can exceed the effect under test.
- Record the exact widget values alongside each image. Otherwise, output from `PP【P】` may
  be attributed to the wrong string.

## 4. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `clip did not tokenize with Anima's dual tokenizer` | The `clip` input is not an Anima checkpoint. Load the Anima model and use its text encoder. |
| `qwen_prefix shares no removable token prefix` | `strip_prefix` is enabled but the prefix is not a token-level prefix of the joined text. Inspect the prefix for leading content, or disable `strip_prefix`. |
| Conditioning is identical to a plain encode | Expected when `qwen_input == t5_input` and `qwen_prefix` is empty: the node then reproduces the upstream single-text path. |
| The generated image contains the prompt as text | The repeated text was placed in `t5_input`. Repetition belongs on the Qwen side. |

## 5. FAQ

**Is a patched ComfyUI required?**
No. The node writes `cross_attn` together with `t5xxl_ids` and `t5xxl_weights` into the
conditioning entry; upstream `Anima.extra_conds` performs the rest.

**Why 512 rows?**
That is upstream's padding target for the adapter output. The node performs no padding;
`Anima.extra_conds` does.

**Can the node be used for negative conditioning?**
It produces one conditioning. A second instance is required if the CFG setting makes the
negative conditioning effective.

**Is a non-Anima model supported?**
No. The mechanism is Anima's dual-tokenizer contract.
