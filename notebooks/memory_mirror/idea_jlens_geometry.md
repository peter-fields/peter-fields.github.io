---
name: idea_jlens_geometry
description: "Research idea — geometric (non-Euclidean / connection) critique of Anthropic's J-lens/J-space, extending Peter's W_QK=G+B non-Euclidean-residual-stream critique. Logged 2026-08-17 for a later project."
metadata: 
  node_type: memory
  type: project
  originSessionId: 24e0f58e-a194-4c1e-a0de-757c991eef02
  modified: 2026-08-17T23:35:26.595Z
---

# Idea: Geometric critique of J-lens / J-space

**Target:** Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models" (transformer-circuits.pub/2026/workspace/) — the **J-lens** interpretability tool and the **J-space** it surfaces (a sparse activation subspace claimed to behave like a Global Workspace).
**Origin:** 2026-08-17, Peter's question connecting his **non-Euclidean residual-stream critique** (W_QK = G + B, see [[idea_qk_metric]]) to J-lens. Parked for a later project. Same critique-family: *an interp method that quietly assumes Euclidean flatness where the model does not.*

## What J-lens / J-space is (from the paper)
- **Averaged Jacobian:** `J_ℓ = E_{t, t'≥t, prompt}[ ∂h_final,t' / ∂h_ℓ,t ]` — first-order effect of perturbing the intermediate residual stream `h_ℓ,t` on the final residual stream `h_final,t'`, averaged over source pos t, target pos t'≥t, and prompts.
- **Lens:** `lens(h_ℓ) = softmax( W_U · norm(J_ℓ h_ℓ) )`  (`norm` = LayerNorm before unembed).
- J-lens vectors = rows of `W_U J_ℓ`; logits = inner products of activations with J-lens vectors.
- **J-space** = sparse non-negative combinations of J-lens vectors (gradient-pursuit sparse decomposition; k vectors reconstruct each activation).

## Geometric analysis — THREE separate questions (don't conflate)
1. **The logit read-out is metric-FREE.** `logit_w = <row_w(W_U), h>` = a covector (row of a linear map V→logits) eating a vector = canonical dual pairing, **no inner product needed**. Contrast QK, which compares *two* residual vectors → needs a bilinear form → the G+B critique. **So the raw lens is NOT where the critique lands.**
2. **The metric critique lands on J-SPACE.** Decomposing `h` into a sparse non-neg combo of J-lens vectors, matched by inner products, treats the J-lens vectors (which are **covectors**, rows of `W_U J_ℓ`) as if they were **vectors** in V, and uses the Euclidean inner product to score reconstruction. Identifying V ≅ V* and measuring reconstruction both **silently pick a metric**. Default Euclidean = coordinate choice = the map/territory point. **REAL HIT.**
3. **Christoffel / connection — the subtle correction to Peter's original phrasing.**
   - A **single first derivative** here (`∂h_final/∂h_ℓ`) is the **differential / pushforward of a map** between spaces. The differential of a *map* needs **NO connection** — Christoffel symbols appear only at *2nd order* (Hessian / second fundamental form ∇dF) or when differentiating a vector *field*. So "the first derivative needs a Christoffel symbol" is **not right as stated**.
   - A **constant non-Euclidean metric G is FLAT** (Γ = 0). G is a Gram matrix → whitens to the identity by a *linear* change of basis. **Non-Euclidean ≠ curved.** So the G critique doesn't invoke Christoffel either.
   - **WHERE the connection genuinely belongs = the AVERAGING.** `J_ℓ = E_t[…]` averages Jacobians computed at **different activations `h_ℓ,t`** = linear maps based at **different points / different tangent spaces**. Averaging tensors at different base points is only legitimate after **parallel transport = a connection = Christoffel symbols**. Flat + global frame → transport trivial → Euclidean average fine. **Curved → not.** And **LayerNorm curves the manifold** (pins activations to a sphere/shell). So the averaged Jacobian assumes flatness illegitimately.

## Sharpened thesis
> J-lens's *averaged* Jacobian AND the J-space reconstruction implicitly assume a **flat Euclidean** residual stream — so that (a) covectors can be read as directions, and (b) Jacobians at different token positions can be summed. But **LayerNorm makes the manifold a sphere** (curved), and the model's own comparison metric is **G ≠ I**. The connection/Christoffel terms belong to the **cross-position averaging on the LayerNorm sphere** — not to any single derivative, nor to a constant G.

## Caveat (lesson from the QK work)
Conceptually valid ≠ empirically large. In the QK study only the **top B modes** were significantly off-embedding; the **bulk sat at the chance baseline**. So *measure the magnitude* of the geometry correction before claiming it matters.

## Possible experiments / next steps
1. **Size the parallel-transport correction** on the LayerNorm sphere: how far apart are the tangent spaces at different activations, really? (sphere radius vs typical activation spread → is the flat-average error 1% or 30%?)
2. **Pre- vs post-LayerNorm averaging frame:** recompute J_ℓ averaging Jacobians in the pre-LN vs post-LN frame — does J-space change? If the "workspace" shifts, part of it is a flat-average artifact.
3. **G-metric (or sphere-metric) J-space:** redo the sparse reconstruction with the model's metric instead of Euclidean — does the recovered J-space / k-vector decomposition change? (mirrors the "is B orthogonal to SAE directions?" test.)
4. **Does the global-workspace claim survive?** Do the 5 functional properties they cite hold for a geometry-aware J-space, or is some of it a Euclidean-averaging artifact?

Related: [[idea_qk_metric]], [[research_ideas]], [[compressed-computation-project]]
