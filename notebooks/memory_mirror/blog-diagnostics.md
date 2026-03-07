# Post 2: Attention Diagnostics — Pre-Experiment Planning (ARCHIVED)
# Post is now complete and live. See post2-experiment-notes.md for actual results.
# Key difference from plan: the signal is in the SHIFT (ΔKL, Δχ) between conditions,
# not in absolute position on the (KL, χ) plane. Original hypothesis below was partially wrong.

## The Diagnostic Quantity

For a given attention head in a given context, let π be the attention weights and z the pre-softmax scores. Introduce temperature T so that:

  π_i(T) = exp(z_i / T) / Σ_j exp(z_j / T)

At T=1, recover actual attention weights. The KL divergence from uniform is:

  D(T) = KL(π(T) ∥ u) = log n − H(π(T))

### Key derivation

  D'(T)|_{T=1} = −Var_π(z)

Steps:
- dπ_i/dT = −(1/T²) π_i (z_i − ⟨z⟩_π)
- log π_i = z_i/T − log Z
- dD/dT = Σ_i (dπ_i/dT) log(nπ_i) = −(1/T²) · β · Var_π(z) ... simplifies to −Var_π(z)/T³
- At T=1: D'(1) = −Var_π(z)

### Critical simplification

Since log π_i = z_i − log Z (constant), we have:

  Var_π(z) = Var_π(log π)

So you only need attention weights, not raw scores. Any model with output_attentions=True works.

## Interpretation of 2D Phase Space (KL, |D'|)

- **High KL, low Var_π(z)**: Certain AND robust. Head is sharply peaked; temperature perturbations don't change behavior. "Knows what it's looking at."
- **Moderate KL, high Var_π(z)**: Somewhat certain but fragile. Scores are spread; slight temperature change reshuffles attention.
- **Low KL, low Var_π(z)**: Near-uniform, insensitive. Head not doing discriminative work in this context.

## Planned Experiment: IOI Circuit

### Setup
- Model: GPT-2-small (12 layers, 12 heads = 144 total)
- Library: TransformerLens
- Data: IOI templates ("When Mary and John went to the store, John gave a drink to")
- Head labels from Wang et al. 2022 (arxiv 2211.00593)

### Monosemantic (circuit) heads
- Name Mover Heads
- Induction Heads
- S-Inhibition Heads
- Backup Name Mover Heads
- Duplicate Token Heads
- Previous Token Heads

### Hypothesis
Circuit heads cluster in high-KL/low-Var corner on IOI inputs.
Non-circuit heads scatter near origin or in high-Var region.

### Follow-up
Run same heads on non-IOI inputs. Circuit heads should migrate toward origin (not activated). Polysemantic heads may maintain moderate KL across both.

## Code Sketch

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

inputs = tokenizer("When Mary and John went to the store, John gave a drink to", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

for layer_idx, layer_attn in enumerate(out.attentions):
    pi = layer_attn[0]  # (heads, seq, seq)
    for h in range(pi.shape[0]):
        w = pi[h, -1, :]  # last token's attention
        log_w = torch.log(w + 1e-12)
        n = w.shape[0]
        kl = (w * (log_w + torch.log(torch.tensor(float(n))))).sum()
        mean_log = (w * log_w).sum()
        var = (w * (log_w - mean_log)**2).sum()
        print(f"L{layer_idx}H{h}: KL={kl:.3f}, Var={var:.3f}")
```

For the real experiment, use TransformerLens for cleaner access to activations and the IOI dataset generator.
