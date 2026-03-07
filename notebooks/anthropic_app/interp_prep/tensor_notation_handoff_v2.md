# Tensor Notation Handoff v2

## Instructions for the Reviewer

Peter has been working through Elhage et al. 2021 ("A Mathematical Framework for Transformer Circuits") and developing a cleaner tensor notation. A first-pass notation was developed and critiqued in a previous session. This document presents the **revised notation** that emerged from that critique, with the key arguments that motivated each choice.

The full paper is at:
`/Users/pfields/Git/peter-fields.github.io/notebooks/anthropic_app/A Mathematical Framework for Transformer Circuits.html`

Relevant lines: notation table at 52771, tensor product definition at 52774–52779, attention head derivation at 6320–6323.

Your job:
1. **Check the notation is internally consistent** — especially index placements and contractions.
2. **Check it against the paper** — does every formula reproduce Elhage's computation?
3. **Scrutinize the arguments** — each notational choice has a motivation; are those motivations sound?
4. **Be critical** — flag anything wrong, incomplete, or overclaimed.

---

## The Core Criticism of Elhage

Elhage's tensor product notation is clean for operators but sloppy for the operand. He defines:

```
(A ⊗ W) x = A x W^T
```

This definition does two things at once:
- Defines how the operator acts
- Encodes a layout convention (x stored with positions as rows)

The W^T is not a mathematical fact about the operation — it is a consequence of storing feature vectors as rows of x rather than columns. Elhage never declares this layout explicitly; it is buried inside the definition of ⊗. The operand x is treated as a bare [n_context, d_model] matrix with no declared tensor product structure, while the operators are written as explicit tensor products. This is an asymmetry.

**The fix**: declare basis vectors explicitly for V_pos and V_feat, write X as an element of V_pos ⊗ V_feat, and let the W^T follow as a consequence rather than a definition.

---

## Basis Declarations

```
e_i : column vector (T×1)       — basis for V_pos
f_μ : row vector (1×d_model)    — basis for V_feat
```

**Why e_i is a column**: standard. Position i is a concrete, labeled thing — token positions are real and discrete. V_pos has a privileged basis; e_i is canonical.

**Why f_μ is a row**: declared by convention, not derived. Feature space has no privileged basis (Elhage says residual stream vectors are "not privileged basis"). The covariant/contravariant distinction on feature indices is therefore conventional bookkeeping, not geometry. We declare f_μ as a row because it reproduces Elhage's [T × d_model] layout and makes the W^T explicit (see below).

**Note**: the asymmetry between e_i (column, canonical) and f_μ (row, conventional) reflects a real asymmetry in the model — positions are semantically meaningful, features are abstract.

---

## The Residual Stream

```
X = X^{iμ} (e_i ⊗ f_μ)
```

e_i is (T×1), f_μ is (1×d_model), their outer product e_i ⊗ f_μ = e_i f_μ is a (T×d_model) rank-1 matrix. Summing X^{iμ} (e_i ⊗ f_μ) over i and μ (Einstein convention: both i and μ are summed since basis vectors carry lower indices) gives a (T×d_model) matrix — exactly Elhage's layout, now declared explicitly.

X^{iμ} are scalar components, both indices upper.

---

## The W^T is Now Explained, Not Defined

With the basis declarations in place:

- **(A ⊗ Id) acting on X**: A left-multiplies the column factor e_i. Result: (AX), positions mixed. No transpose.
- **(Id ⊗ W) acting on X**: W is defined to left-multiply column vectors in V_feat. But f_μ is a **row**. So W must right-multiply as W^T.

```
(Id ⊗ W) X = X W^T
```

This is now a **consequence** of declaring f_μ as a row — not a primitive definition. Elhage's W^T is explained.

Full action:
```
(A ⊗ W_OV) X = A X W_OV^T
```

---

## Index Conventions

- **i, j** : position indices (range over T = n_context)
- **μ, ν** : feature indices (range over d_model)
- **Upper index** : contravariant — component of X
- **Lower index** : covariant — basis vector index, or "input" slot of an operator
- **Einstein contraction** : one index upper, one lower

---

## The Operators

**A (attention pattern):**
```
A^i_j   shape [T, T]
```
Destination position i (upper), source position j (lower). For fixed i, A^i_j is a distribution over source positions j. Used in the contraction R^{iμ} = A^i_j V^{jμ} where j contracts upper (on V) against lower (on A).

**W_OV (combined value + output projection):**
```
(W_OV)^μ_ν   shape [d_model, d_model],   W_OV = W_O W_V
```
A (1,1) tensor — linear map on V_feat. Takes contravariant feature input ν, produces contravariant feature output μ.

**W_QK (combined query-key metric):**
```
(W_QK)_μν   shape [d_model, d_model],   W_QK = W_Q^T W_K
```
A (0,2) tensor — bilinear form on V_feat. Takes two contravariant feature vectors, produces a scalar attention score.

**Key point**: W_OV and W_QK are the same shape but genuinely different algebraic objects. W_OV is a linear map (1,1); W_QK is a bilinear form (0,2). Elhage's notation lists both as "[d_model, d_model] matrices" with no indication of this distinction. The index placement makes it explicit.

**Why W_QK is (0,2)**: The dot product q·k = (W_Q x_i)·(W_K x_j) uses the Euclidean metric δ_{ab} on head space to contract the d_head indices:
```
(W_QK)_μν = δ_{ab} (W_Q)^a_μ (W_K)^b_ν = Σ_a (W_Q)^a_μ (W_K)^a_ν
```
The Euclidean metric on ℝ^{d_head} is what makes W_QK a (0,2) form rather than a (1,1) map. The transpose in W_Q^T W_K is hiding this metric contraction.

---

## Full Attention Computation

Step by step:
```
V^{iμ} = (W_V)^μ_ν X^{iν}              — apply W_V to each token
R^{iμ} = A^i_j V^{jμ}                  — mix positions (j contracts upper/lower)
h^{iμ} = (W_O)^μ_ν R^{iν}             — apply W_O to each token
```

Collapsed:
```
h^{iμ} = A^i_j (W_OV)^μ_ν X^{jν}
```

j contracts (upper on X, lower on A). ν contracts (upper on X, lower on W_OV). i and μ are free — same index structure as X, so h can be added back to the residual stream.

**Equivalence with Elhage**: Elhage writes h(x) = (A ⊗ W_OV) x = A x W_OV^T. In components:
```
h(x)_{iμ} = Σ_j Σ_ν A_{ij} x_{jν} (W_OV)_{μν}
           = A^i_j (W_OV)^μ_ν X^{jν}   ✓
```
The W_OV^T in Elhage is absorbed into the index placement — (W_OV)^μ_ν has μ upper (output) and ν lower (input), so contracting with X^{jν} (ν upper) automatically implements the transpose. No explicit ^T needed.

---

## Attention Score — Two Steps

The attention score computation involves a subtlety: the softmax changes the index placement of j.

**Pre-softmax scores** (j is a free upper index on both inputs):
```
s^{ij} = X^{iμ} (W_QK)_μν X^{jν} / sqrt(d_k)
```
μ contracts (upper on X^{iμ}, lower on W_QK). ν contracts (upper on X^{jν}, lower on W_QK). i and j are both free and both upper.

**Softmax** (normalizes over j for fixed i, changes j from upper to lower):
```
A^i_j = softmax_j( s^{ij} )
```

This two-step presentation is important. The `~` shorthand `A^i_j ~ softmax( X^{iμ} (W_QK)_μν X^{jν} )` conceals an inconsistency: j is upper on the RHS but lower on the LHS. The softmax is the one place in the computation where a non-tensor (nonlinear) operation also changes the index placement of a free index. Writing it as two steps makes this explicit.

---

## What This Notation Does vs. Elhage

### Improvements
- X is explicitly declared as an element of V_pos ⊗ V_feat — no implicit layout convention
- W^T follows from the declaration of f_μ as a row, rather than being baked into the definition of ⊗
- W_OV (1,1) and W_QK (0,2) are distinguished as different algebraic types despite identical shape
- The hidden Euclidean metric on head space in W_QK is surfaced
- The softmax index placement change is made explicit

### What Elhage Does Better
- The mixed-product property (A⊗B)(C⊗D) = (AC)⊗(BD) is visually immediate in Elhage's notation and makes virtual attention heads `(A^{h2}A^{h1}) ⊗ (W_OV^{h2}W_OV^{h1})` obvious
- The factored structure — that position mixing and feature mixing are independent operations — is more visible in the ⊗ notation than in index expressions
- Elhage's notation is more compact for multi-layer path expansions

The two notations are complementary. The index notation is more honest about types and contractions; the ⊗ notation is more transparent about structure.

---

## Key Claims to Scrutinize

1. Is the argument that f_μ being a row *explains* W^T actually correct? Or does it just restate the same convention in different language?

2. Is the two-step softmax presentation (s^{ij} upper, A^i_j lower) the right way to handle the index placement change? Or is there a cleaner way?

3. The claim that W_QK is (0,2) because of the Euclidean metric on head space — is this the right way to think about it, or is there a cleaner story? Does this metric play any role elsewhere in the model?

4. Is the asymmetry between e_i (canonical column) and f_μ (conventional row) well-motivated? Or does it import a spurious distinction?

5. Is anything lost by not having an explicit e^i (dual position basis) or f^μ (dual feature basis)? Where would those appear if the notation were extended?