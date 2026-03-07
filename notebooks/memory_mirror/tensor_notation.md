# Tensor Notation for Elhage et al. 2021

## Context
Developed while studying "A Mathematical Framework for Transformer Circuits" for Anthropic interview prep. The paper's tensor product notation is correct but never declares the operand x — this is the source of all confusion.

## The One Declaration That Fixes Everything

> Residual stream vectors x_i live in V_feat* (dual/covector space), not V_feat.

Everything else follows from standard tensor calculus.

## The Notation

```
x = X_μ^i (e_i ⊗ f^μ)   ∈  V_pos ⊗ V_feat*
```

- e_i ∈ V_pos: canonical column basis, lower i (positions are labeled, privileged basis)
- f^μ ∈ V_feat*: dual basis covectors, upper μ (model chose freely, no privileged basis)
- X_μ^i: scalar components, lower μ (Greek, feature), upper i (Roman, position label)
- Bilinearity of ⊗ lets X_μ^i move freely: e_i ⊗ (X_μ^i f^μ) = X_μ^i (e_i ⊗ f^μ) ✓
- Einstein sum on i (Roman) is valid here — orthogonality of {e_i} protects components from mixing. The sum builds the full residual stream from T rank-1 contributions. NO caveat about crossing ⊗ boundary needed.

## Index Conventions

- **Greek (μ, ν)**: Einstein summation, upper = contravariant/row, lower = covariant/column. Both geometric meaning AND layout convention — they reinforce each other.
- **Roman (i, j)**: Einstein summation allowed. Upper = row vector, lower = column vector — layout convention only, NO covariant/contravariant content (V_pos has canonical basis, no basis changes to worry about).
- The Latin/Greek split does heavy bookkeeping work: Greek = feature space algebraic machinery, Roman = position labels.

## Tensor Types

| Object | Type | Space | Meaning |
|---|---|---|---|
| e_i | column, lower Roman | V_pos | canonical position basis |
| f^μ | row, upper Greek | V_feat* | dual feature basis |
| x^i = X_μ^i f^μ | covector | V_feat* | token i's representation (row vector) |
| A^i_j | (1,1) | V_pos | attention pattern: upper i = destination row, lower j = source column |
| (W_OV^T)* | (1,1) | V_feat* | correct operator on covectors; W_OV^T appears explicitly in index formula |
| W_QK | (2,0) | V_feat* | bilinear form on covectors — like a metric on V_feat* (asymmetric, indefinite) |

## Why x_i is a Covector

Features are linear functionals — they fire (produce a scalar) when a certain pattern is present. The model learned to turn each token into a linear function on feature space, in whatever basis minimized loss (hence no privileged basis). Row vectors ARE covectors. The PyTorch convention [batch, seq, d_model] with d_model last reflects this — features in last dim, tokens as rows, d_model axis = Greek feature index = always axis=-1.

Additional interpretability value:
- SAE encoder rows ∈ V_feat* (read), decoder columns ∈ V_feat (write) — read/write asymmetry falls out naturally
- W_QK as (2,0) explains why QK and OV circuits are fundamentally different algebraic objects, not just two matrices
- Superposition: features are linear functionals packed into V_feat* at non-orthogonal angles — no constraint forces orthogonality since there's no canonical inner product on V_feat*
- Explains why you can't take dot products of residual stream vectors and call them "similarity" — no metric on V_feat*; W_QK provides one per head, but it's learned not canonical

## The W^T Explained

Elhage writes h(x) = AxW_OV^T. The W^T encodes that the correct abstract operator on V_feat* is (W_OV^T)* (dual of the transpose), not W_OV*:
- W_OV* in row form: x → x W_OV (no transpose) — WRONG
- (W_OV^T)* in row form: x → x W_OV^T — correct ✓

The ^T and * are NOT two separate operations that cancel. (W_OV^T)* is a single map on V_feat* whose matrix in row form is W_OV^T. The "cancellation" is only at the level of which matrix acts on components — (W_OV^T)* in column form acts as M (same matrix as W_OV), but this doesn't mean the two operations cancel in any deeper sense.

## The Correct Index Formula

The full attention head output, derived by applying (A ⊗ (W_OV^T)*) to x = X_μ^i (e_i ⊗ f^μ):

```
h(x) = (A ⊗ (W_OV^T)*) · X_μ^i (e_i ⊗ f^μ)
      = X_μ^i A^k_i (W_OV^T)^μ_ν (e_k ⊗ f^ν)
```

Step by step:
- Linearity: X_μ^i · (A ⊗ (W_OV^T)*)(e_i ⊗ f^μ)
- A(e_i) = A^k_i e_k (Einstein on i, output position k)
- (W_OV^T)*(f^μ) = (W_OV^T)^μ_ν f^ν (Einstein on ν — wait, ν is output, not summed. μ upper on (W_OV^T) contracts with... actually μ is the label of the dual basis input, ν is the output index. See below.)

Einstein checks in h_ν^k = A^k_i X_μ^i (W_OV^T)^μ_ν:
- i: upper on X, lower on A → proper Einstein ✓
- μ: lower on X, upper on (W_OV^T)^μ_ν → proper Einstein ✓
- k: free upper (output position)
- ν: free lower (output feature)

IMPORTANT: the formula uses (W_OV^T)^μ_ν, NOT (W_OV)^μ_ν. The transpose is explicit in the index formula. (W_OV^T)^μ_ν = M^T[μ,ν] = M[ν,μ].

Mapping to Elhage: h[k,ν] = Σ_i Σ_μ A[k,i] x[i,μ] (W_OV^T)[μ,ν] = [AXW_OV^T][k,ν] ✓

Output h_ν^k lives in V_pos ⊗ V_feat* — same type as x. Residual stream addition is legal.

## W_QK Notation

```
s^{ij} = X_μ^i (W_QK)^{μν} X_ν^j
```

- μ: query-side feature index (contracts with query token i)
- ν: key-side feature index (contracts with key token j)
- (W_QK)^{μν}: upper μ = query index (row of W_QK matrix), upper ν = key index (column)
- W_QK plays role of a (asymmetric, indefinite) metric on V_feat* — pairs two covectors to give a scalar

## Softmax Index Change

```
s^{ij}  — both i,j upper: equal footing in index placement (before softmax)
A^i_j   — j lowered: source becomes column index in probability distribution
r^i = A^i_j v^j  — Einstein on j (lower on A, upper on v^j): weighted sum of row vectors
```

Note on "symmetric in type": s^{ij} has both i,j upper — equal placement. This does NOT mean the matrix is symmetric (W_QK is not generally symmetric). It means source and destination are in the same notational position before softmax acts.

Note: r^i = A^i_j v^j is a **deviation from Elhage**, who writes r_i = Σ_j A_{i,j} v_j (flat subscripts, explicit sum). Computation identical; our version encodes the softmax type change and uses Einstein on j.

"j is now column" after softmax: yes — j was a row-vector position index (upper), now it's a column index in a distribution (lower). The index lowering is a notational convention tracking the semantic change, not a tensor operation (softmax is nonlinear).

## Connection to SVD / Schmidt Decomposition

x = X_μ^i (e_i ⊗ f^μ) is the standard basis expansion of an element of V_pos ⊗ V_feat*. SVD gives the minimal decomposition (rank(X) terms). Schmidt rank = matrix rank. "Entangled" residual stream = rank > 1 = positions and features don't factorize. The standard basis expansion uses T terms (one per token).

## Python Compatibility

The formula h_ν^k = A^k_i X_μ^i (W_OV^T)^μ_ν maps directly to h = A @ x @ W_OV^T in PyTorch:
- x is [seq, d_model] — tokens as rows, d_model last
- A is [seq, seq]
- W_OV^T is [d_model, d_model]
- Greek feature index ν is always last axis (axis=-1) — consistent with d_model last convention
- Roman position index k is second-to-last (seq axis)

## What Elhage Gets Right

The mixed product property (A⊗W)(A'⊗W') = (AA')⊗(WW') is algebraic — survives all of this. Circuit composition, virtual heads, QK/OV separation all intact. We added honesty about what x is; we didn't touch the circuit machinery.

## Deviations from Elhage Notation

1. x declared as element of V_pos ⊗ V_feat* (Elhage: bare matrix)
2. r^i = A^i_j v^j with index placement encoding type (Elhage: r_i = Σ_j A_{i,j} v_j flat)
3. W_QK explicitly (2,0), W_OV explicitly (1,1) — Elhage lists both as "[d×d] matrices"
4. Softmax index lowering j upper → lower made explicit
5. Everything else matches Elhage exactly

## Bug in Elhage (verified against paper HTML)

Elhage's attention score formula (line 6323 of HTML) reads:
  A = softmax(x^T W_Q^T W_K x)

But his notation table (line 52771) declares x^n as shape [n_context, d_model] — rows are tokens.
With that layout, x^T W_Q^T W_K x has mismatched inner dimensions and does NOT produce [n_context, n_context].

The formula only works if x is column-stacked [d_model, n_context]. But h(x) = A x W_OV^T (line 52779) uses rows-as-tokens. The two formulas use opposite conventions.

Correct row-convention formula: A = softmax( x W_QK x^T / sqrt(d_k) )
Our index notation: s^{ij} = X^{iμ} (W_QK)_μν X^{jν} → [x W_QK x^T]_{ij} ✓ — convention-free and correct.

## Blog Idea: Symmetric/Antisymmetric Decomposition of W_QK

W_QK = W_QK^sym + W_QK^anti  where  W_QK^sym = (W_QK + W_QK^T)/2,  W_QK^anti = (W_QK - W_QK^T)/2

- s^{ij}_sym = X^{iμ} (W_QK^sym)_μν X^{jν} is symmetric: s^{ij}_sym = s^{ji}_sym → mutual relevance / content matching
- s^{ij}_anti = X^{iμ} (W_QK^anti)_μν X^{jν} is antisymmetric: s^{ij}_anti = -s^{ji}_anti → directional routing

Interpretability prediction: ‖W_QK^anti‖/‖W_QK^sym‖ is a per-head scalar that should discriminate:
- Content-matching heads (name mover, duplicate token): low ratio — symmetric component dominates
- Positional/directional heads (prev-token, induction): high ratio — antisymmetric component dominates

Testable against Wang et al. 2022 IOI head labels. No precedent found in the literature as of Mar 2026.
Even with symmetric W_QK, attention is still directional due to per-row softmax — but the antisymmetric
part adds an ADDITIONAL directional bias beyond what softmax normalization provides.
