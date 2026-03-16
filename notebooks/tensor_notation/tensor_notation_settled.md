# Tensor Notation for Elhage et al. 2021 — Settled Version

**This is the authoritative notation file. Earlier files (handoff.md, handoff_v2.md) are superseded.**

---

## The Core Declaration

```
x = X^i_μ (e_i ⊗ f^μ)  ∈  V_pos ⊗ V_feat*
```

- `e_i ∈ V_pos`: column vector (T×1), canonical basis for position space. Lower Roman index.
- `f^μ ∈ V_feat*`: row vector (1×d_model), dual basis for feature space. Upper Greek index.
- `X^i_μ`: scalar components. Upper Roman i (which position), lower Greek μ (which dual basis element).

**Python:** `e_i` is shape [T, 1], `f_mu` is shape [1, d_model]. Their outer product `e_i @ f_mu` is [T, d_model] — a rank-1 matrix. Summing `X^i_μ (e_i ⊗ f^μ)` over i and μ builds the full [T, d_model] residual stream. The tensor product structure and the Python array structure are the same object.

---

## Index Conventions

- **Roman (i, j, k, l)**: position indices, range over n_context. Upper = row label, lower = column label. No covariant/contravariant content — V_pos has a canonical basis, no basis changes.
- **Roman (v, w, x, y, z)**: vocab indices, range over n_vocab. Same upper/lower convention as position indices. V_vocab has a canonical basis (one dim per token); no primal/dual content.
- **Greek (μ, ν, ...)**: d_model feature indices (middle of Greek alphabet). Upper = primal (V_feat), lower = dual/covariant (V_feat*). Einstein summation: contracts when one upper + one lower.
- **Greek (α, β, γ, δ; θ, η if needed)**: d_head indices (beginning of Greek alphabet). Same upper/lower convention as d_model indices.
- **r, s**: singular component labels (Roman), used in SVD decompositions.

**Alphabetical = destination before source.** For any index pair, the alphabetically earlier letter is the destination. This holds for both position indices and head labels:
- Position: (A)^i_j — i is destination, j is source (i comes before j)
- Heads: g = query-side head (destination), h = key-side head (source)

**Layer and head labels** are written inside parentheses as superscripts — they are labels, not contractable indices:
- (x^l)^i_μ — residual stream after layer l, position i, d_model index μ
- (A^{l,h})^i_j — attention pattern, layer l, head h, destination i, source j
- (W_OV^{l,h})^μ_ν — OV matrix, layer l, head h
- (W_Q^{l,h})^μ_α, (W_K^{l,h})^μ_α, (W_V^{l,h})^μ_α, (W_O^{l,h})^α_μ — Q/K/V/O projections

---

## Key Objects and Their Types

| Object | Index form | Type | Notes |
|---|---|---|---|
| Residual stream components | X^i_μ | scalars, (1,1) tensor | upper Roman = position, lower Greek = dual feature |
| Position basis | e_i | column vector, lower Roman | V_pos, canonical |
| Feature dual basis | f^μ | row vector, upper Greek | V_feat*, conventional |
| Attention pattern | A^k_i | (1,1) on V_pos | upper k = destination, lower i = source |
| OV operator on V_feat* | (W_OV^T)^μ_ν | (1,1) on V_feat* | upper μ = input dual index, lower ν = output |
| QK bilinear form | (W_QK)^{μν} | (2,0) on V_feat* | both upper: eats two covectors, gives scalar |
| Embedding matrix | (W_E)^w_μ | V_vocab → V_feat* | upper w = vocab input, lower μ = feat output covector |
| Unembedding matrix | (W_U)^μ_w | V_feat* → V_vocab* | upper μ contracts with residual stream, lower w = vocab logit |
| Logits | l^i_w = (x^l)^i_μ (W_U)^μ_w | row covector in V_vocab* | lower w indexes vocab; softmax over w gives next-token probs |

---

## The Attention Head — OV Circuit

```
h^k_ν = A^k_i X^i_μ (W_OV^T)^μ_ν
```

**Contractions:**
- i: upper on X, lower on A — sum over source positions ✓
- μ: lower on X, upper on (W_OV^T) — canonical dual pairing ✓
- Free: k upper (output position), ν lower (output feature)

Output h^k_ν has the same index structure as X^k_ν — legal to add to residual stream ✓

**Python:** `h = A @ X @ W_OV_T` with shapes [T,T] @ [T,d] @ [d,d] = [T,d] ✓

---

## Why W_OV^T and Not W_OV

W_OV is standardly defined as a linear map V_feat → V_feat (acts on column vectors). We need a map V_feat* → V_feat* (acts on row vectors / covectors). Two options:

- **W_OV*** (dual of W_OV): acts on row vectors as x → x W_OV — **wrong**, gives AXW_OV not AXW_OV^T
- **(W_OV^T)*** (dual of the transpose): acts on row vectors as x → x W_OV^T — **correct** ✓

The explicit W_OV^T in the index formula `(W_OV^T)^μ_ν` is the honest statement that we're applying the dual of the transpose. The W^T is a consequence of x living in V_feat*, not a definition.

**Interpretive content:** (W_OV^T)* transforms *which linear functional* the residual stream is measuring. Source j measures certain feature directions; the head writes a new linear functional to destination k. It's a re-sensing operation.

---

## The Attention Head — QK Circuit

**Pre-softmax scores** (both i, j free upper — equal footing):

```
s^{ij} = X^i_μ (W_QK)^{μν} X^j_ν / sqrt(d_k)
```

Contractions: μ (lower on X^i, upper on W_QK), ν (lower on X^j, upper on W_QK). Both i, j free and upper ✓

**Softmax:** A^i_j = softmax_j(s^{ij}) — j is lowered (source becomes column index in a probability distribution over sources).

**Why W_QK is (2,0):** it must eat two covectors X^i_μ and X^j_ν (both lower Greek). It therefore needs two upper Greek indices. This makes W_QK a bilinear form on V_feat* — a learned, per-head "similarity structure" on the dual feature space. There is no canonical inner product on V_feat*; W_QK provides one per head, explicitly.

---

## The Covector Interpretation

Declaring x ∈ V_pos ⊗ V_feat* has genuine interpretive content:

Each token's representation x^i = X^i_μ f^μ is a **linear functional on V_feat**. It assigns a real number to any primal feature direction v ∈ V_feat:

```
x^i(v) = X^i_μ v^μ
```

"Feature f is active at position i" = X^i_μ v^μ_f is large — canonical dual pairing, **no metric required**.

This makes explicit that V_feat* has no preferred inner product. Taking dot products of residual stream vectors and calling them "similarity" imports an arbitrary Euclidean metric. W_QK provides the principled version: a learned bilinear form making attention similarity explicit and per-head.

**SAE read/write:** encoder rows ∈ V_feat* (detect features via duality pairing, same space as x); decoder columns ∈ V_feat (primal write directions). The asymmetry is the natural primal/dual structure.

*Caveat: computationally, V_feat* = ℝ^{d_model} and the pairing is just a dot product. The value is conceptual clarity about which operations require a metric and which don't.*

---

## Connection to Elhage

Elhage's formula `h(x) = (A ⊗ W_OV) x = A x W_OV^T` is correct. His definition `(A ⊗ W) x = A x W^T` encodes the row-vector layout convention implicitly — the W^T is buried in the definition of ⊗ rather than made explicit by the index structure of x.

**Changes from Elhage:**
1. x declared as element of V_pos ⊗ V_feat* — layout is explicit, not implicit
2. W^T follows as a consequence of f^μ being a row basis vector — not a primitive definition
3. W_QK explicitly (2,0), W_OV explicitly (1,1) — Elhage lists both as [d_model × d_model] matrices
4. Softmax index change j upper → lower made explicit
5. The mixed-product property (A⊗B)(C⊗D) = (AC)⊗(BD) is untouched — circuit algebra intact

---

## Multi-Layer Circuit Notation

**Residual stream update after layer 1:**

```
(x^2)^j_μ = (x^1)^j_μ + Σ_h (h^{1,h})^j_μ
```

where the head output is:

```
(h^{1,h})^j_μ = (A^{1,h})^j_k (x^1)^k_ν (W_OV^{1,h})^ν_μ
```

**Layer 2 query and key (single head):**

```
q^i_α = (x^2)^i_μ (W_Q^{2,h})^μ_α

k^j_α = (x^2)^j_μ (W_K^{2,h})^μ_α
```

**K-composition key** (layer 1 OV feeds into layer 2 W_K — the interesting term):

```
k^j_α = (x^1)^j_μ (W_K^{2,h})^μ_α  +  (A^{1,h})^j_k (x^1)^k_ν (W_OV^{1,h})^ν_μ (W_K^{2,h})^μ_α
         direct path                     K-composition term
```

**Layer 2 attention pattern (K-composition term):**

```
(A^{2,h})^i_j = softmax_j( (x^2)^i_μ (W_Q^{2,h})^μ_α · (A^{1,h})^j_k (x^1)^k_ν (W_OV^{1,h})^ν_μ (W_K^{2,h})^μ_α )
```

Information flow reads directly off indices: i ← j ← k (destination ← key position ← source of key content).

**Induction head:** (A^{1,h})^j_k ≈ δ^j_{k+1} (prev-token head), so k^j contains content from position j-1. Layer 2 fires when query at i matches token j-1 — i.e. when the same token appeared earlier in the sequence.

**Joint Q-K composition** (both sides enriched by layer 1, different heads g and h):

```
(A^{2})^i_j = softmax_j( (A^{1,g})^i_k (x^1)^k_ν (W_OV^{1,g})^ν_μ (W_Q^2)^μ_α · (A^{1,h})^j_l (x^1)^l_ρ (W_OV^{1,h})^ρ_σ (W_K^2)^σ_α )
```

Head labels: g (query-side, alphabetically before h) and h (key-side) — consistent with destination-before-source convention.

---

**The Elhage convention inconsistency:** Elhage's attention score formula (HTML line 6323) reads `A = softmax(x^T W_Q^T W_K x)` using column convention, while `h(x) = Ax W_OV^T` (HTML line 52779) uses row convention. The row-convention form is `A = softmax(x W_QK x^T / sqrt(d_k))`. Our notation is convention-free and consistent throughout.