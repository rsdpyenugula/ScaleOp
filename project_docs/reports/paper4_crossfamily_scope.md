# Cross-family conversion: compatibility test + what ScaleOp needs to fan out

Scope for Paper 4. Paper 1 established that conversion value rides on the
**representation channel** (within-family held-out ridge R² = 0.84 across sizes,
while parameters align weakly). Everything below follows from one question: does
that channel stay open across *families*, and if so, what has to change in the
code and in the method?

---

## 1. The compatibility gate (run this first)

`experiments/m6_crossfamily.py` measures whether a donor family's representations
are a linear image of a target family's, using the same method as Paper 1's §4 so
the numbers are directly comparable.

```bash
# control: reproduce the within-family number (expect ridge R^2 ~0.84)
uv run python experiments/m6_crossfamily.py --donor 1.4b --target 410m
# cross-family probes
uv run python experiments/m6_crossfamily.py --donor 1.4b --target Qwen/Qwen2.5-0.5B
uv run python experiments/m6_crossfamily.py --donor 1.4b --target allenai/OLMo-2-0425-1B
```

Two obstacles are handled inside the probe:

- **Different tokenizers** → each model tokenizes the *same documents* with its own
  tokenizer, and we compare **one mean-pooled vector per document per layer**.
  Pooling over the sequence is tokenization-invariant: row *i* of both matrices
  summarizes the same text, regardless of how many tokens each family used.
- **Different depths** → we align at matched *relative* depth (target layer *j* vs
  donor layer `round(j·(Ld−1)/(Lt−1))`) and also dump the all-pairs CKA matrix.

**Decision rule.** Let *r* be mean ridge R² across matched depths.

| *r* vs the 0.84 within-family reference | Reading | Action |
|---|---|---|
| **r ≳ 0.7** | channel is open; geometry is largely shared | proceed to weight-space conversion |
| **0.4 ≲ r ≲ 0.7** | partial alignment | conversion plausible but expect a much larger recovery budget; report the gap as the result |
| **r ≲ 0.4** | families do not share a linear geometry | selection-based conversion cannot work; the *negative* result is the contribution |

Also report **ridge − Procrustes**: a large gap means the relationship needs a
general linear map (scaling/shearing of feature axes), not just a rotation, which
is evidence about *how* the families differ, not just how much.

---

## 2. Blocker A — the tokenizer, and how to sidestep it

Different tokenizers mean the embedding matrix and the output head are
**not transferable by selection at all**: row *k* of the donor's `EMB_IN` is a
different token than row *k* of the target's. This is not a small loss — for
Pythia-410M, `EMB_IN` + `EMB_OUT` are ~103M of 410M parameters (≈25%).

Three options, in order of research cleanliness:

1. **Keep the donor's tokenizer (recommended first step).** We are *minting a new
   model*, so we may choose its tokenizer. Converting Pythia-6.9B → a 2B Llama-shaped
   target that still uses Pythia's tokenizer isolates the **architecture** question
   from the **vocabulary** question. One variable at a time, consistent with the
   rest of the project.
2. **Vocab-overlap initialization.** Copy rows for tokens whose surface strings
   match, initialize the rest from the mean embedding. Standard in the
   tokenizer-transplant literature; adds a confound but tests a realistic setting.
3. **Learned vocab bridge.** Fit a map between embedding spaces from co-occurrence.
   A research problem in its own right — out of scope for Paper 4's first pass.

**Consequence for the paper:** report cross-family conversion under (1) as the main
result, and (2) as the "you must also change tokenizer" ablation. Never mix them.

---

## 3. Blocker B — wiring differences, and what each one breaks

Paper 1's two levers are defined by *which paths a normalization fronts*.
Different families change that map, so the safe-wiring analysis must be redone.

| | Pythia (GPT-NeoX) | Llama / Qwen2.5 / OLMo-2 |
|---|---|---|
| Norm | LayerNorm (weight **and** bias), parallel attn+MLP | RMSNorm (weight only), sequential |
| MLP | GELU, `up` → `down` (2 matrices) | **SwiGLU**: `gate`,`up` → `down` (3 matrices) |
| Attention | MHA, **fused** QKV, partial rotary (`rotary_pct`=0.25) | **GQA** (KV heads < Q heads), separate q/k/v, full rotary |
| Embeddings | untied | often **tied** at small sizes |

What this does to the method:

- **Compensation (function lever) — survives, with care.** It needs a purely linear
  path with no normalization between cut and read. Llama/Qwen still have exactly
  two: `o_proj` and `down_proj`. But SwiGLU's `down_proj` reads the *elementwise
  product* `silu(gate)·up`, so the second-moment Σ must be measured **at
  `down_proj`'s input** (post-gating), not at the MLP input. Hook placement, not
  new math.
- **Rescale (dynamics lever) — extends naturally.** The argument is that a
  downstream norm renormalizes each rescaled read-in path, so the scalar cancels in
  the forward pass but changes the variance the optimizer sees. RMSNorm
  renormalizes too (it just skips mean-subtraction), so the argument holds; under
  SwiGLU **both** `gate` and `up` are norm-fronted and therefore rescale-eligible.
- **Whole-head selection — needs a real change for GQA.** KV heads are shared by
  groups of Q heads. Dropping a KV head means dropping its **entire group**, so
  head selection must score and drop at *group* granularity. Ignoring this
  silently corrupts attention.
- **Rotary selection — needs a change.** Paper 1 keeps rotary pairs at stride 2 to
  preserve `(cos,sin)` pairing, derived for `rotary_pct`=0.25. With full rotary the
  frequency ladder is the whole head dim, so the stride rule must be recomputed
  from the target's `head_dim` and base θ.
- **Tied embeddings.** If the target ties `EMB_IN`/`EMB_OUT`, they cannot be
  compensated/rescaled independently; the code must detect and respect tying.

---

## 4. Blocker C — code generalization (the actual work)

The good news: extraction already speaks a **role-keyed dict** —
`{(layer, "MLP_UP"): W, (-1,"EMB_IN"): W, ...}` — which is the right seam. Fanning
out means writing per-family adapters that produce/consume the same roles, not
rewriting the method.

Current architecture coupling (`grep` count of NeoX-specific references):

| File | refs | Change needed |
|---|--:|---|
| `lib/models.py` | 20 | `MODEL_SUITE` is Pythia-only and `extract_weights` reads `gpt_neox.*` / fused QKV. Needs a per-family extractor keyed off `config.model_type`. |
| `lib/assemble.py` | 15 | Builds `GPTNeoXForCausalLM` from roles. Needs a per-family builder. |
| `lib/subclone.py` | 9 | Selection assumes fused QKV, MHA, `rotary_pct`, 2-matrix MLP. Needs GQA-group and SwiGLU handling. |
| `lib/activations.py` | 4 | Hardcodes `model.gpt_neox.layers` (already flagged in a comment). `experiments/m6_crossfamily.py:block_list()` has the resolver to lift. |

**Proposed abstraction: a `WiringSpec` per family**, declaring
(i) how to find blocks, (ii) role → module path, (iii) which roles are
norm-fronted (rescale-eligible), (iv) which are LN-free linear read-outs
(compensation-eligible), (v) attention grouping (MHA vs GQA), (vi) rotary layout,
(vii) whether embeddings are tied. Then `subclone.py` becomes family-agnostic and
each new family is one declarative spec plus a test.

New roles to add: `MLP_GATE` (SwiGLU), and KV-group metadata for GQA.

---

## 5. Staged plan with gates

1. **Gate 1 — representation probe** (`m6_crossfamily.py`, hours, no training).
   Run the within-family control plus 2–3 target families. If *r* ≲ 0.4 everywhere,
   Paper 4 becomes a negative result and stops here — cheaply, which is the point
   of running this first.
2. **Gate 2 — `WiringSpec` + one new family, same tokenizer.** Implement the spec
   abstraction and port to Qwen or OLMo (fully open; Llama is gated). Verify with
   the **bit-exact reconstruction control** Paper 1 already uses: rebuild the donor
   from its own extracted roles and require identical logits. This single test
   catches nearly every porting bug.
3. **Gate 3 — within-family conversion in the new family.** Reproduce Paper 1's
   ladder (random / selection / +compensation / +rescale) inside Qwen or OLMo. This
   is also exactly the experiment ICLR reviewers want for **Paper 1's** revision —
   do it once, use it twice.
4. **Gate 4 — true cross-family** (Pythia donor → Qwen-shaped target, donor
   tokenizer retained), then the tokenizer-change ablation.

**Honest risk assessment.** Gate 3 is a solid, publishable contribution on its own
("the two-lever decomposition ports across architectures, and here is the
per-architecture safe-wiring analysis"). Gate 4 is the ambitious claim and may fail
on representation mismatch — which is why Gate 1 runs first and costs nothing.
