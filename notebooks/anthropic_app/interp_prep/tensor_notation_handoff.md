# Tensor Notation Handoff

## Instructions for the Reviewer

The user (Peter) has been reading Elhage et al. 2021 ("A Mathematical Framework for Transformer Circuits") and working through their tensor product notation. The full paper is available at:

`/Users/pfields/Git/peter-fields.github.io/notebooks/anthropic_app/A Mathematical Framework for Transformer Circuits.html`

The tensor product notation section is at line 52774. The key definition is at line 52779:

```
(A ⊗ W) x = A x W^T
```

Peter found Elhage's notation slightly sloppy — the object x is never declared as a proper tensor, the ⊗ symbol is overloaded, and the transpose in W^T is hidden inside the definition of ⊗ rather than made explicit via index structure.

A proposed cleaner notation was developed during conversation. Your job is to:

1. **Scrutinize this notation carefully** — is it internally consistent? Is it compatible with Elhage's definitions? Does it actually clean things up or just add complexity?
2. **Check it against the HTML** — read the relevant sections of the paper and verify the proposed notation is saying the same thing.
3. **Be critical** — flag anything that is wrong, incomplete, or misleading. Do not just validate.
4. **Explain to Peter** what is going on clearly, using the notation that is most honest and illuminating.

---

## The Proposed Notation

### Index conventions

- **i, j** : position indices (token positions, range over T)
- **μ, ν** : feature indices (model dimensions, range over d_model)
- **Upper index** : contravariant — "output" or "destination"
- **Lower index** : covariant — "input" or "source"
- **Contraction** : occurs when the same index appears once upper, once lower (Einstein convention). Two upper indices do NOT contract.

### The residual stream

X lives in V_pos ⊗ V_feat, with components:

```
X^i^μ
```

where i is the position index (upper, contravariant) and μ is the feature index (upper, contravariant). Both upper because X is the thing being written to / output of the residual stream.

X can be decomposed as:

```
X^i^μ = e^i ⊗ x^i^μ
```

where e^i is the i-th basis vector in position space (contravariant, upper) and x^i^μ is the feature vector at position i. The i on x^i^μ is a free index (same position), not a contraction — both are upper so Einstein convention does not trigger a sum.

### The operators

**A (attention pattern):**
```
A^i_j   shape (T, T)
```
Destination position i (upper), source position j (lower). A^i_j is the attention weight from destination i to source j. Summing A^i_j over j (contraction) mixes source positions into destination position i.

**W_V (value projection):**
```
(W_V)^μ_ν   shape (d_model, d_head) in standard convention
```
Maps input feature ν (lower) to output feature μ (upper).

**W_O (output projection):**
```
(W_O)^μ_ν   shape (d_head, d_model) in standard convention
```
Maps input feature ν (lower) to output feature μ (upper).

**W_OV = W_O W_V (combined):**
```
(W_OV)^μ_ν = (W_O)^μ_ρ (W_V)^ρ_ν
```
ρ is a contracted (dummy) intermediate feature index.

**W_QK (attention score metric):**
```
(W_QK)_μν   shape (d_model, d_model)
```
Both indices lower — this is a (0,2) tensor, i.e. a bilinear form / metric on V_feat. It pairs the query feature vector with the key feature vector to produce a scalar attention score. Analogous to g_μν in GR.

### Full attention computation

Step by step:

```
V^i^μ = (W_V)^μ_ν X^i^ν          (apply W_V to each token)
R^i^μ = A^i_j V^j^μ               (mix positions — j contracts upper/lower)
h^i^μ = (W_O)^μ_ν R^i^ν           (apply W_O to each token)
```

Collapsed:

```
h^i^μ = A^i_j (W_OV)^μ_ν X^j^ν
```

j contracts (upper on X, lower on A), ν contracts (upper on X, lower on W_OV). i and μ are free — same index structure as X, so h can be added back to X in the residual stream.

### Connection to Elhage's notation

Elhage writes:

```
h(x) = (A ⊗ W_OV) · x
```

with the definition `(A ⊗ W) x = A x W^T`.

In our notation this is:

```
h^i^μ = A^i_j (W_OV)^μ_ν X^j^ν
```

The W^T in Elhage's definition is absorbed into the index placement — W_OV has output index μ upper and input index ν lower, so the contraction with X^j^ν (upper ν) is automatically the transpose operation. No explicit ^T needed.

### The metric / raising and lowering

W_QK plays the role of the metric on V_feat:

```
A^i_j ~ softmax( X^i^μ (W_QK)_μν X^j^ν / sqrt(d_k) )
```

(W_QK)_μν lowers the feature index of X^i^μ to produce a covariant vector, which then contracts with X^j^ν (upper ν on the key side). This is the only place where feature indices are lowered. The residual stream X^i^μ always has both indices upper.

There is no natural metric on V_pos, so position indices cannot be freely raised/lowered.

---

## Key claims to verify / scrutinize

1. Is `h^i^μ = A^i_j (W_OV)^μ_ν X^j^ν` actually equivalent to `A x W_OV^T` in Elhage's sense? Check the W^T carefully.

2. Is it correct that both position and feature indices on X are upper? Or should one be lower? Check against the residual stream addition `X → X + h`.

3. Is W_QK really a (0,2) metric tensor? Or is it better thought of as a linear map? Check the attention score formula.

4. Does the e^i ⊗ x^i^μ decomposition work properly? Is the double upper i actually unambiguous or does it create confusion?

5. Is there anything this notation makes *harder* to see compared to Elhage's original? Don't just look for what it clarifies — look for what it obscures.
