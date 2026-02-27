---
title: "Attention Diagnostics: Testing KL and Susceptibility on the IOI Circuit"
layout: single
author_profile: false
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
sidebar:
  - title: "Notation"
    text: |
      **Attention**<br>
      \\(\pi\\) — attention weights<br>
      \\(z\\) — pre-softmax scores<br>
      \\(n\\) — sequence length<br>
      \\(u = 1/n\\) — uniform distribution<br>
      <br>
      **Diagnostics**<br>
      \\(\hat{\rho}\_{\text{eff}}\\) — \\(\text{KL}(\hat{\pi} \| u)\\)<br>
      \\(\chi\\) — \\(\text{Var}\_{\hat{\pi}}(\log \hat{\pi}) / (\log n)^2\\)<br>
      \\(\Delta\text{KL}\\) — shift between conditions<br>
      \\(\Delta\chi\\) — shift between conditions<br>
tags: [mechanistic-interpretability, attention, transformers, IOI-circuit, diagnostics]
excerpt: "The previous post introduced KL selectivity and susceptibility χ as per-head diagnostics derivable from attention weights alone. Here I test them on GPT-2-small's IOI circuit: can two scalar statistics, computed from a single forward pass, distinguish the 23 known circuit heads from the other 121? It seems so!"
---

<!--
=== POST PLAN (revised Feb 24) ===

FIGURES & FLOW:

Fig 1: One Name Mover head (e.g. L9H9), non-IOI (gray) vs IOI (red) in (KL, χ) plane.
   Arrow from inactive mean → active mean. This is the hook — directly tests
   the Post 1 prediction. "The circuit fires, and we can see it in these diagnostics."

Fig 2: Big grid of ALL 23 circuit heads, same format. Organized by role.
   Shows fingerprints — heads with same role cluster in same region.
   Selection heads shift right (more selective). Structural heads don't move.
   S-Inhibition mixed. Role information lives in both position AND shift direction.

Fig 3: Grid of ~10 non-circuit heads for comparison.
   Pick heads that span the range: a few boring ones (near-zero ΔKL, as expected),
   plus one or two with surprisingly large shifts (L9H4, L10H3) to hint that
   not all non-circuit heads are inert. Contrast with Fig 2.

Report: Circuit vs non-circuit statistical test.
   |ΔKL| p=0.0002, |Δχ| p<0.0001. The manipulation is minimal (one name repeats
   or doesn't), yet circuit heads clearly feel it.

Fig 4: KL vs χ scatter for all 144 heads on IOI prompts.
   corr = 0.34. They measure different things. Don't belabor — just show it.
   LEAVE OUT the ΔKL-Δχ correlation (r=0.70). Looks less clean in the scatter
   than the number suggests — too many heads where one changes but the other
   doesn't. Too nuanced for this post.

Fig 5 (teaser): One correlation analysis plot.
   Maybe dendrogram-ordered C_diff matrix with circuit heads marked.
   Or the C_IOI circuit-only 23×23 block. Just enough to say "there's more
   structure here" and point to a future post.

KEY POINT TO MAKE: These diagnostics only need forward passes — no gradients,
no activation patching, no model modification. They could work on models where
per-component causal intervention is prohibitively expensive. That's the "so what"
beyond validating a known circuit.

LIMITATIONS (be honest):
   - One circuit, one model. No generalization claim.
   - how could it be useful when we don't know the circuit a priori
   - Layer depth confound: corr(layer, KL) ≈ 0.38.
   - KL is blind to target identity — captures *how selectively* a head attends,
     not *what* it attends to. A head that redirects attention without changing
     selectivity is invisible to ΔKL. Same for χ.

=== NOTEBOOK PLAN (separate curated notebook) ===

1. Define KL and χ from Post 1's theory
   - Brief recap, link back to blog
   - KL(π̂ ∥ u) measures selectivity. χ = Var_π(log π)/(log n)² measures susceptibility.

2. Setup: GPT-2-small, IOI circuit, 50/50 matched prompts
   - 144 heads. IOI vs non-IOI (third-name-C design). Exactly 15 tokens both sets.
   - Head labels from Wang et al. 2022.

3. Circuit heads respond more than non-circuit heads
   - |ΔKL|: circuit vs non-circuit p=0.0002. |Δχ|: p<0.0001.

4. Selection heads become more selective; structural heads don't care
   - ΔKL positive for NMs/Backup NMs/Neg NMs. ΔKL ≈ 0 for structural.

5. KL and χ measure different things but respond together
   - corr(KL, χ) = 0.34. corr(ΔKL, Δχ) = 0.70.

6. The (KL, χ) fingerprint figure
   - Role clustering in absolute position.

7. Limitations
   - One circuit, one model. Layer depth confound (corr ≈ 0.38).
   - KL blind to target identity (shape not content).
   - Polysemantic interpretation of χ not demonstrated.

Cut from notebook: ⟨z⟩ excess, criticality framing, stat mech analogy.
v1→v2→v3 prompt iteration: brief collapsible section, not main narrative.
-->

[Post 1]({% post_url 2026-02-17-why-softmax %}) derived two per-head diagnostics from the structure of the softmax operator: **KL selectivity** \\(\hat\rho\_{\text{eff}} = \text{KL}(\hat\pi \\| u)\\) measures how sharply a head focuses its attention, and **susceptibility** \\(\chi = \text{Var}\_{\hat\pi}(\log\hat\pi)/(\log n)^2\\) measures how sensitive that sharpness is to small changes in the query-key scores. Both are computed from a single forward pass — no gradients, no activation patching.

Here I test a concrete prediction: *circuit heads should show larger shifts in these diagnostics between activating and non-activating prompts than non-circuit heads.* The testbed is GPT-2-small's Indirect Object Identification (IOI) circuit,[^wang2022] whose 23 heads and functional roles are well characterized.

---

## Setup

### Recap

A **prompt** is a token sequence \\(x = (x_1, \ldots, x_n)\\). GPT-2-small processes it layer by layer, maintaining a **residual stream** \\(h_i^{(l)} \in \mathbb{R}^{d_{\textrm{model}}}\\) for each position — a contextualized representation that accumulates the contributions of all attention heads and MLPs up to layer \\(l\\).

Each of the 144 heads (12 layers × 12 heads) projects the residual stream into queries and keys, computes scores \\(z_{ij} = q_i \cdot k_j / \sqrt{d_k}\\), and applies softmax to produce an **attention distribution** over source positions:

$$\hat{\pi}_j = \text{softmax}(z_j).$$

This \\(\hat{\pi}\\) depends on both \\(x\\) (through the queries and keys) and the head's learned parameters. From it we read off the two diagnostics introduced in [Post 1]({% post_url 2026-02-17-why-softmax %}):

$$\hat{\rho}_{\textrm{eff}} = \frac{\text{KL}(\hat{\pi} \| u)}{\log n}, \qquad \chi = \frac{\text{Var}_{\hat{\pi}}(\log \hat{\pi})}{(\log n)^2},$$

where \\(u = (1/n, \ldots, 1/n)\\) is the uniform distribution, and we have normalized each by prompt length. No backward pass needed — \\(\hat{\pi}\\) is already computed in the forward pass.

### The experiment

GPT-2 small is known to have a circuit that performs indirect object identification. [^wang2022] Let's say we have a prompt of 15 tokens that reads:

$$
x = \textrm{When Alice and Bob went to the store, Bob gave a drink to ___}.
\label{eq:good_prompt}
$$

The correct next token the model should predict is Alice (the indirect object of the second clause.) It turns out that Wang et al. showed that several heads perform various jobs in order to predict this next token correctly. One head detects a name has appeared twice, another suppresses the name that has appeared twice, another moves the name that appeared once, and so forth.

For our experiments, we shall generate 50 IOI prompts of length \\(n=15\\) of exactly the same format as \\eqref{eq:good_prompt} along with 50 non-IOI prompts that have a similar format but no repeating name, e.g.

$$
x = \textrm{After Mary and John sat down for dinner, Sarah gave a gift to ___}
$$

which should not "activate" the circuit. 

---

## A single circuit head

The hook. L9H9 is a Name Mover — one of the heads most directly responsible for copying the indirect object token to the output. On IOI prompts (where the circuit should fire) versus matched non-IOI controls (where it should not):

{% include figure image_path="/assets/images/posts/attention-diagnostics/fig1_L9H9_name_mover.png" alt="L9H9 Name Mover — IOI vs non-IOI in (KL, χ) plane" caption="**Figure 1.** L9H9 (Name Mover) in the \\((\hat\\rho\_\\text{eff},\\,\\chi)\\) plane. Red: IOI prompts (circuit active). Gray: non-IOI prompts (circuit inactive). Stars mark condition means; arrow shows the shift." %}

[...]

---

## All 23 circuit heads

{% include figure image_path="/assets/images/posts/attention-diagnostics/fig2_all_circuit_heads.png" alt="All 23 IOI circuit heads — active vs inactive" caption="**Figure 2.** All 23 IOI circuit heads organized by role. Each panel shows per-prompt \\((\\hat\\rho\_\\text{eff},\\,\\chi)\\) for IOI (colored) and non-IOI (gray). Arrow: non-IOI mean → IOI mean." %}

[...]

---

## Non-circuit heads

{% include figure image_path="/assets/images/posts/attention-diagnostics/fig3_non_circuit_heads.png" alt="Selected non-circuit heads for comparison" caption="**Figure 3.** Selected non-circuit heads. Orange (★): top-4 by \\(|\\Delta\\text{KL}|\\). Blue: 12 inert heads with near-zero shift. Most non-circuit heads are indifferent to whether the IOI circuit fires." %}

[...]

---

## Statistical test: do the diagnostics separate circuit from non-circuit?

[...]

{% include figure image_path="/assets/images/posts/attention-diagnostics/fig4_delta_distributions.png" alt="|ΔKL| and |Δχ| distributions for circuit vs non-circuit heads" caption="**Figure 4.** Distributions of \\(|\\Delta\\text{KL}|\\) (left) and \\(|\\Delta\\chi|\\) (right) for circuit (red) vs non-circuit (gray) heads. Mann-Whitney U: \\(p=0.00020\\) and \\(p<0.0001\\) respectively." %}

[...]

---

## Cross-head correlations: a teaser

The diagnostics so far treat heads independently. But heads in the same functional role don't just shift in isolation — they tend to move together. Computing the 144×144 Pearson correlation matrix of KL selectivity across prompts reveals block structure aligned with the circuit.

{% include figure image_path="/assets/images/posts/attention-diagnostics/fig5_corr_matrices.png" alt="Cross-head KL correlation matrices C_IOI, C_nonIOI, C_diff" caption="**Figure 5.** Cross-head KL correlation matrices. \\(C\_\\text{IOI}\\) and \\(C\_\\text{non-IOI}\\) are 144×144 Pearson correlation matrices computed across 50 prompts each; \\(C\_\\text{diff} = C\_\\text{IOI} - C\_\\text{non-IOI}\\) isolates task-specific coupling. Brown margin ticks mark the 23 known circuit heads." %}

{% include figure image_path="/assets/images/posts/attention-diagnostics/fig6_nc_circuit_scatter.png" alt="NC-circuit coupling scatter" caption="**Figure 6.** Every (NC head, other head) pair plotted as \\((C\_\\text{IOI},\\, C\_\\text{non-IOI})\\). Points on the diagonal have \\(C\_\\text{diff}=0\\). The gray band is \\(\\pm 5\\sigma\\) of the empirical \\(C\_\\text{diff}\\) distribution. Blue: NC head paired with a circuit head; red: two NC heads. Labeled: the three NC heads with the largest \\(|C\_\\text{diff}|\\) coupling to a circuit head." %}

[...]

---

## Limitations

[...]

---

## What's next

[...]

---

[^wang2022]: Wang et al. (2022). [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593).