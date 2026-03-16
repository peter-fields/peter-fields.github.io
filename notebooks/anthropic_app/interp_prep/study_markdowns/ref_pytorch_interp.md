# PyTorch / TransformerLens / SAE / Circuits Reference

---

## PyTorch Basics

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tensors
x = torch.tensor([[1., 2.], [3., 4.]])
x = torch.zeros(3, 4)
x = torch.randn(3, 4)
x.shape                          # torch.Size([3, 4])

# Ops (same as numpy, different names)
x @ y                            # matmul
x.T                              # transpose
torch.einsum("ij,jk->ik", x, y)
torch.softmax(x, dim=-1)         # dim = axis in numpy
torch.triu(torch.ones(T, T), diagonal=1)   # causal mask

# Indexing, reshape, permute
x[0, :]                          # first row
x.reshape(T, -1)
x.permute(0, 2, 1)               # reorder dims (like np.transpose with axes)

# Convert to/from numpy
x.detach().numpy()               # tensor → numpy
torch.from_numpy(arr)            # numpy → tensor

# Device
x = x.to("cuda")                 # GPU (if available)
x = x.to("cpu")
```

### nn.functional (F) — stateless math ops
```python
F.softmax(x, dim=-1)
F.cross_entropy(logits, targets)   # targets = integer class indices
F.relu(x)
F.layer_norm(x, normalized_shape=[d_model])
F.linear(x, W, b)                 # x @ W.T + b  (note: W.T, not W)
```

### Building a trainable model
```python
class MyModel(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out)   # learnable W + bias

    def forward(self, x):
        return self.W(x)

model = MyModel(16, 4)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in data:
    optimizer.zero_grad()          # MUST clear grads each step
    loss = F.cross_entropy(model(x), y)
    loss.backward()                # compute gradients
    optimizer.step()               # update weights
```

---

## TransformerLens Basics

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")

# Config
model.cfg.d_model      # 768 for gpt2
model.cfg.n_heads      # 12
model.cfg.n_layers     # 12
model.cfg.d_head       # 64
model.cfg.d_vocab      # 50257

# Weights (all torch tensors)
model.W_E              # [d_vocab, d_model]   embedding
model.W_U              # [d_model, d_vocab]   unembedding
model.W_Q              # [n_layers, n_heads, d_model, d_head]
model.W_K              # [n_layers, n_heads, d_model, d_head]
model.W_V              # [n_layers, n_heads, d_model, d_head]
model.W_O              # [n_layers, n_heads, d_head, d_model]

# Circuit matrices for layer l, head h
W_QK = model.W_Q[l, h] @ model.W_K[l, h].T   # [d_model, d_model]
W_OV = model.W_V[l, h] @ model.W_O[l, h]     # [d_model, d_model]

# Tokenization
tokens = model.to_tokens("Mary and John went")   # [1, T] int tensor
strs = model.to_str_tokens(tokens)               # list of strings

# Forward pass + activation cache
logits, cache = model.run_with_cache(tokens)
# logits: [batch, T, d_vocab]

# Common cache keys
cache["pattern", layer]        # attention weights [n_heads, T, T]
cache["resid_pre", layer]      # residual stream before layer [T, d_model]
cache["resid_post", layer]     # residual stream after layer
cache["z", layer]              # value outputs before W_O [T, n_heads, d_head]
cache["result", layer]         # head outputs after W_O [T, n_heads, d_model]
```

---

## SAEs — Sparse Autoencoders

**Concept:** learn an overcomplete dictionary of features such that residual stream activations are sparse linear combinations of dictionary vectors.

```
x ≈ W_dec @ f(W_enc @ x + b_enc) + b_dec
f = ReLU  (or JumpReLU)
```

- `W_enc`: [d_model, d_features] — encoder (reads from stream)
- `W_dec`: [d_features, d_model] — decoder (writes back)
- `d_features >> d_model` (e.g. 4x–16x overcomplete)
- Loss = reconstruction + sparsity: `||x - x_hat||^2 + λ ||f||_1`

**NumPy-only implementation:**
```python
def sae_forward(x, W_enc, b_enc, W_dec, b_dec):
    # x: (d_model,)
    pre = x @ W_enc + b_enc          # (d_features,)
    f = np.maximum(pre, 0)           # ReLU → sparse activations
    x_hat = f @ W_dec + b_dec        # (d_model,)
    return x_hat, f

def sae_loss(x, x_hat, f, lam=1e-3):
    recon = np.mean((x - x_hat)**2)
    sparsity = lam * np.sum(np.abs(f))
    return recon + sparsity
```

**With TransformerLens** (using a pretrained SAE):
```python
# SAE libraries: transformer_lens + sae_lens or nnsight
# Typical usage: get residual stream activations, pass through SAE
resid = cache["resid_post", layer]           # [T, d_model]
x_hat, f = sae_forward(resid[token_pos], *sae_weights)
top_features = np.argsort(f)[::-1][:10]     # top active features
```

**Key ideas:**
- Features = columns of W_dec (dictionary atoms)
- Active features at a position = which neurons fire for that token
- Monosemantic: each feature ideally corresponds to one interpretable concept
- Dead features: neurons that never activate (training problem)

---

## Causal Interventions — IOI Circuit (Wang et al. 2022)

### Task
"When Mary and John went to the store, John gave a drink to ___"
Model should predict "Mary" (IO = indirect object) over "John" (S = subject).

### Metric — logit difference
```python
def logit_diff(logits, io_token, s_token):
    # logits: [1, T, d_vocab]
    return (logits[0, -1, io_token] - logits[0, -1, s_token]).item()
```
Higher = model correctly predicts IO over S.

### Corrupted input
Swap the names: "When John and Mary went to the store, Mary gave a drink to ___"
Now the model should predict "John" — logit diff goes negative.

### Activation patching
Restore one component at a time from clean run into corrupted run.
If logit diff recovers → that component is causally important.

```python
logits_clean, cache_clean = model.run_with_cache(clean_tokens)
logits_corrupt, cache_corrupt = model.run_with_cache(corrupt_tokens)

def make_patch_hook(clean_val):
    def hook(value, hook):
        value[:] = clean_val
        return value
    return hook

# Patch residual stream at layer l
logits_patched = model.run_with_hooks(
    corrupt_tokens,
    fwd_hooks=[(f"blocks.{l}.hook_resid_pre",
                make_patch_hook(cache_clean["resid_pre", l]))]
)
ld = logit_diff(logits_patched, io_token, s_token)
```

### Mean ablation
Replace activation with mean over many inputs (cleaner baseline than zero):
```python
mean_val = cache_clean["resid_pre", l].mean(dim=0, keepdim=True)

def ablate_hook(value, hook):
    value[:] = mean_val
    return value
```

### Path patching
Patch a specific *edge* (A → B), not a full node.
Isolates: "how much does head A's output causally affect head B specifically?"

Procedure:
1. Run clean → save output of A
2. Run corrupted → at B's input, splice in A's clean output only
3. Everything else stays corrupted

### Direct logit attribution (linear, no patching needed)
```python
# Contribution of head h at layer l to logit diff
head_out = cache["result", layer][:, h, :]        # [T, d_model]
direction = model.W_U[:, io_token] - model.W_U[:, s_token]   # [d_model]
contribution = head_out[0, -1] @ direction         # scalar
```
Decomposes logit diff = sum of contributions from all heads + MLPs (exact, no approximation).

### What the IOI circuit found
Key head types discovered by systematic patching:
- **Name Mover heads** (layers 9-10): copy IO token to output position
- **Duplicate Token heads** (early layers): detect when S appears twice
- **S-Inhibition heads** (layer 8): suppress S token prediction
- **Induction heads**: general in-context copying mechanism

Circuit = the subgraph of heads + paths sufficient to explain IOI behavior.
Found by: patch all heads, rank by importance, then verify with ablation.
