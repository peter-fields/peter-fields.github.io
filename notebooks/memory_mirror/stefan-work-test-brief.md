---
name: stefan-work-test-brief
description: "Quick-reference brief for the Pivotal/Stefan Heimersheim 2h work test (due Mon May 18 6:59 AM CDT) — task constraints, CC mechanics, red-team checklist, what Claude should and shouldn't do during the timer. Reload at start of any work-test session."
metadata: 
  node_type: memory
  type: project
  originSessionId: 12de4699-df4e-4626-83b9-1af439944490
---

# Stefan/Pivotal Work Task — Context Brief

> **STATUS: SUBMITTED 2026-05-18 ~5:44 AM CDT.** PDF writeup at `~/Downloads/peter fields work trial for stefan.pdf`, Colab link in the doc. Waiting for Stefan's response. This file is retained as reference for what was learned + what to follow up on if Stefan asks.
>
> **Key empirical results delivered:**
> - Task 1: model memorizes 100% accuracy across n_pairs sweep up to 8192 with d_mlp=128. **Bigram baseline (embed @ unembed alone) = 48%** — half the memorization lives outside the MLP.
> - Task 2: top-1 accuracy by category (capital best at 45%, several categories at 0%). LogitLens shows **two-phase emergence**: rank narrows smoothly across layers 0-7, logprob jumps sharply at L7→L8 (+4.71 mean). **Empirical confound discovery**: 9 of 57 "correct" capitals emerge at layer 0 because their answers begin with " the".
>
> **What was underdeveloped in the writeup (due to time):**
> - Task 1 Q3: only gestured at "K > N setup" + "randomize embed/unembed". The fuller answer about sparse multi-feature inputs, multi-feature outputs, forcing non-linearity to do work, etc. was not written up.
> - Task 2 Q3 confounds: only " the"-prefix confound and a single sentence on distributed-vs-localized. Template uniformity, famous-bias, LogitLens limitations, selection bias not addressed.
> - Task 2 Q4 (better methods like activation patching, probing): not addressed at all.
> - The "effective rank" framing got one sentence ("correlated variables decrease effective number of variables") — the deepest single insight but underdeveloped.
>
> *Reload at start of any work-test session. Working dir: `~/Git/peter-fields.github.io/notebooks/stefan_work_test/` with subdirs `notebook_downloaded/`, `scratch_notebooks/`, `scratch_figs/`, `final_notebook_to_submit/`, `final_figs/`.*

## Task constraints
- **Deadline**: Mon May 18, 6:59 AM CDT (AoE Sun May 17)
- 2h timer, **TWO tasks** (task 2 has more complex code per Stefan)
- GPT-2 + TransformerLens (verified: loads in 2.5s on Peter's MPS, fwd+cache in 1.6s)
- **AI allowed for code mods; NOT for writing the report** (Stefan's explicit advice)
- Skills assessed: **red-teaming, experiment design, attention to detail, presentation, thoroughness**
- Output: 1–3 page report, mostly figures (Colab screenshots fine)

## Stefan Heimersheim (Pivotal mentor for this assessment)
- Core author of APD paper (parameter-space decomposition, Apollo Research)
- Coauthor of "Compressed Computation is (probably) not CiS" LW critique
- Values directness; willing to debunk own field's claimed results
- Explicit quote: *"I found reports with significant LLM-contribution worse than those without"*

## CiS vs CC — the key conceptual distinction
- **CiS (Computation in Superposition)** = a *mechanism*: sparsity + superposition + ReLU noise cleanup. Requires that only a few features are active per input so neurons can be reused across input subsets.
- **CC (Compressed Computation)** = an *outcome*: "more functions than neurons," regardless of mechanism.
- APD authors deliberately picked "compressed computation" because they suspected their toy model achieved the outcome WITHOUT the CiS mechanism (page 11 of APD paper).
- The critique then confirmed: CC ≠ CiS. CC's gain comes from exploiting interference structure of fixed random W_E, not from sparsity-based reuse.
- **Hanni definition of CiS** needs BOTH: (a) inputs AND outputs stored in superposition, (b) more computations than non-linearities.
- Field status: CiS never cleanly demonstrated in toy models that survive scrutiny; **zero direct evidence in LLMs**.

## CC architecture (if task uses this setup)
- **Pipeline**: 100 → 1000 → 50 → 1000 → 100
  - Input x ∈ ℝ^100, sparsely active, x_i ∈ [-1, 1]
  - W_E shape (1000, 100): embeds into residual stream. **Columns are unit-norm 1000-D feature directions** (paper says "unit norm rows" but means columns; W_E[:,i] is the embedding direction for feature i)
  - Residual stream is 1000-D
  - MLP: **20× down-projection to 50 ReLU neurons, then back up** (opposite of GPT-2's 4× up-projection)
  - W_U = W_E^T shape (100, 1000): unembeds; **tied weights** (read-out direction for output i = embedding direction for feature i)
- **Target function**: y_i = x_i + ReLU(x_i), per output i
- **Why d_resid = 1000 (post-hoc)**: chosen so model beats the monosemantic baseline. Small d_resid → more interference between feature columns → worse residual bypass for linear part of y.
- **Why W_E fixed (load-bearing)**: prevents trivial alignment (model can't reshape W_E to align features with neurons → monosemantic baseline). But also creates the exploitable interference pattern W_E^T @ W_E - I that the critique flags.

## Neuron contribution metric (APD paper, used to visualize polysemantic structure)
For input feature i, neuron k:
```
contribution_i[k] = (W_U[i,:] · W_OUT[:,k]) × (W_IN[k,:] · W_E[:,i])
                     └── output-side ──┘     └── input-side ───┘
```
Full formula: `(W_U[i,:] @ W_OUT) ⊙ (W_IN @ W_E[:,i])` → 50-D vector.
- Input-side term: pre-activation of neuron k if x = e_i (one-hot feature i)
- Output-side term: how much a unit of neuron k's activation translates into output i
- Element-wise product enforces AND: neuron must both read AND write for the feature
- Caveat: this is static (depends only on weights), doesn't account for ReLU non-linearity or simultaneous-feature interference

## Red-team checklist (apply to any toy model claiming CiS)
**Generic moves (CC critique template):**
1. **Does the gain survive when you remove the obvious shortcut?**
   - CC example: set M=0 (clean labels). Result: gain vanished.
2. **Does performance scale with size of the shortcut?**
   - CC: loss scaled linearly with |M|. Smoking gun.
3. **Can you construct an alternative non-CiS solution that matches?**
   - CC: SNMF of M achieves comparable loss with zero ReLU computation.
4. **Where do learned parameters concentrate?**
   - CC: neuron directions in the 50-D positive-eigenvalue subspace of M.
5. **Is the baseline fair?**
   - CC: naive 0.0833; model 0.06; but only because of M.
6. **Does result survive distribution shift?**
   - Generic: train on one distribution, test on another. Real CiS should transfer.

**CC-specific hooks (if the task hands you this architecture):**
7. **Vary W_E's random seed.** Re-train with different W_E samples; check if "gain over baseline" is consistent or seed-dependent. High variance → exploitation of specific noise structure.
8. **Vary d_resid.** Train with d_resid = 200, 500, 1000, 2000. Plot gain over baseline vs d_resid. The post-hoc choice of 1000 is suspicious; gain should NOT depend on this if CiS is real.
9. **Project learned W_in/W_out onto bases.** Two candidate bases: (a) eigenvectors of W_E^T @ W_E (interference basis), (b) columns of W_E (feature basis). Concentration on interference basis = noise-fitting. Concentration on feature basis = genuine per-feature mechanisms.
10. **Compare to SNMF benchmark.** Compute the SNMF approximation of the relevant matrix; if its loss matches the trained CC model, the "computation" claim is hollow.
11. **Check vary the residual bypass.** Zero out the residual connection around the MLP and re-evaluate. If gain over baseline disappears, the MLP isn't doing CiS — the residual is doing most of the work.
12. **Test input sparsity dependence.** Vary input sparsity (1, 3, 5, 10, 30 active features per input). True CiS should depend strongly on sparsity; CC's noise-fitting mechanism should be less sparsity-dependent.

## How to test "neurons live in top-k subspace of M" (Finding 2, critique paper's actual methodology — Figure 7)
A two-figure template:

**Figure 7a — per-neuron cosine-similarity heatmap.**
Four-panel grid:
- Rows: top panels use **eigenvectors of M**; bottom panels use **singular vectors of M+I**
- Columns: left uses W_in (read directions); right uses W_out (write directions)
- Each panel: cosine_similarity(neuron direction, eigenvector/SV), as a heatmap
- X-axis: vector index, sorted by eigenvalue/σ descending. Y-axis: neuron index. Color: cosine sim.
- Expected pattern: bright band in cols 0–49, black in cols 50–99.
- Use **cosine similarity** (normalized), NOT raw dot product — controls for neuron norm.

**Figure 7b — ReLU-free MLP projection test (conceptually cleanest single statement).**
For each eigenvector or singular vector v of M (or M+I):
1. Project through MLP's linear-equivalent: `v' = W_out @ W_in @ v`
2. Compute cosine_sim(v, v')
3. Plot cosine_sim vs vector index

Logic: `W_out @ W_in` has rank ≤ 50 (bottleneck), so preserves at most 50 dimensions. Which 50?
- If v lies in the rank-50 image → v' approximately parallel to v → cos_sim ≈ 1
- If v orthogonal to image → v' ≈ 0 → cos_sim ≈ 0

Expected: cos_sim ≈ 1 for top 50; drops to ≈ 0 for bottom 50.

**Why both M and M+I?** Earlier finding established MLP's effective linear op is `W_out @ W_in ≈ M + c·I` (M for mixing, c·I for linearized ReLU). So:
- Eigenvectors of M alone → align with the M part of the MLP
- SVs of M+I → align with the **combined** M + c·I → matches what MLP actually does, slightly stronger alignment
- For M alone they use eigendecomposition focused on positive-eigenvalue subspace (M may not be symmetric)
- For M+I they use SVD

**Practical:**
- Do BOTH W_in (read) and W_out (write) — claim needs both.
- Sanity baseline: same analysis on randomly-init or M=0-trained network. Trained-CC should align; baseline shouldn't.
- Plot σ spectrum of M FIRST. If values fall off slowly (no clear "top 50"), the claim is structurally weaker.
- Get the right M. The critique's reformulation makes M tunable (embedding-based, random, M=0). For original CC setup, M is effectively determined by W_E^T @ W_E plus the explicit identity from y = x + ReLU(x).

## Open questions Stefan flagged at the end of the critique (target these in the work test)
From the critique's conclusion: "we have not fully reverse-engineered the compressed computation model... we think an analysis of how the solution relates to the eigenvectors of M, and how it changes for different choices of M are very promising."

**Two specific open questions Stefan calls "promising":**

1. **How does the trained solution relate to M's eigenstructure (beyond top-50 alignment)?**
   - Established: read/write directions concentrate in top-50 SVs of M+I
   - Established: SNMF of M+I partially replicates the loss
   - Open: trained model has additional sparsity and correlation-with-M structure not captured by SNMF. What is it?
   - Open: do specific neurons map to specific eigenvectors, or is it diffuse? What's the per-neuron-per-eigenvector structure?

2. **How does the solution change for different choices of M?**
   - Established: loss scales with |M|
   - Open: what about M's rank? Condition number? Symmetry/skew-symmetry? Sparsity? Block structure?
   - Open: at what point does the trained model stop being "SNMF-like" and become something else?

**Stefan's note: "DM / email StefanHex" — he is actively soliciting follow-up.** A report that produces a new piece of analysis on one of these questions has more value than re-confirming the existing findings.

## Why CC matters for real LLMs (the eigenstructure-encoding hypothesis)
The CC critique uncovers a mechanism that may be the real phenomenon in LLM MLPs — *not* CiS in the Hanni sense, but **structured-interference-exploitation via low-rank linear operations**.

**What CC's MLP turned out to be doing:**
- Low-rank linear approximation of M (mixing matrix)
- Neuron directions concentrate in top-50 eigenvectors/SVs of M
- ReLU contributes ≈ a single constant offset (almost no per-feature non-linearity)
- SNMF of M+I plugged in as weights replicates much of this

**Translated to LLMs:**
- Residual streams have rich structured interference from token co-occurrence + LayerNorm + attention
- An MLP doing CC-style work would: align neurons with top eigenvectors of activation covariance; be mostly linear; look polysemantic at the neuron level (each eigenvector mixes many human-named features)
- This matches: hard-to-interpret neurons, SAE feature splitting, single-direction ablation failures, the elusive "Paris neuron"

**Implication: the CiS framing may be wrong for real LLMs.**
- Not "false" (math is fine), but "looking for the wrong mechanism"
- LLM compression (Claude knows SF streets) may be eigenstructure-encoding, not sparsity-based-reuse
- Tools that look for CiS (Hanni-style) might miss what MLPs actually do
- APD components might correspond to eigenstructure rather than computations
- "Facts > neurons" intuition still holds, but mechanism is "facts stored as linear combinations of eigenvectors," not "sparse packed computations"

**There may be MULTIPLE phenomena conflated as "CiS":**
- Genuine sparsity-based-reuse (Hanni's CiS, possibly rare)
- Eigenstructure-encoding (CC-style, possibly common)
- Memorization (Geva key-value)
- Each requires different detection tools; conflating them is what makes interp hard

**Concrete research moves downstream of this:**
- Test eigenstructure hypothesis in a real LLM MLP: compute eigenstructure of residual stream activations on a corpus, check if neuron directions align with top eigenvectors
- Compare what SAEs find to what eigendecomposition finds; check if SAE features are reconstructable from eigenstructure
- Build a separate toy model exhibiting eigenstructure-encoding by construction (validates eigenstructure-detection tools)

## Effective-rank reframing (the unifying principle)
The CC critique is a special case of a general principle: **when features are correlated, you only need as many neurons as the *effective rank* of the task, not the *nominal rank*.**

- CC nominally: 100 functions, 50 neurons → looks like 2× compression → looks like CiS
- CC actually: M has effective rank ≤ 50 (the top-50 SVs dominate), so 50 neurons suffice trivially. No "compression" beyond exploiting natural rank reduction.

This generalizes:
- **"LLMs represent more concepts than dimensions"** = concepts are correlated; effective rank of conceptual subdomain < total dim. This is Mikolov-style covariance structure, not CiS.
- **"MLPs compute more functions than neurons"** = the useful functions are correlated; effective rank of "what's needed at layer L" is small.
- **"SAEs find atomic features"** = they may be finding eigenvectors of residual stream covariance. Feature splitting is the diagnostic — no privileged basis means SAE size determines decomposition.

**Hanni's CiS framework is mathematically valid but possibly mis-applied.** The boring effective-rank explanation may suffice for everything observed. CiS would require **compression beyond effective rank** — that's the discriminating test.

**Discriminating test for "true CiS vs effective-rank compression":**
1. Compute effective rank R of the relevant matrix/task (e.g., M's σ spectrum, or activation covariance's eigenspectrum)
2. Predict the loss/accuracy achievable by rank-R linear approximation
3. Compare to trained model's actual performance
4. If trained model ≈ rank-R prediction → effective-rank compression, no CiS
5. If trained model significantly outperforms → genuine CiS-like contribution beyond rank

The CC critique implicitly does step 4 by showing SNMF(M+I) matches the trained model. SNMF is rank-50 linear approximation. Trained CC ≈ SNMF → no compression beyond effective rank → not CiS.

**For the work test**: if you compute M's effective rank and predict what rank-R approximation should give, and compare to the trained model, you've done the cleanest version of the analysis Stefan is asking for in his open questions.

## Quick SNMF reference
SNMF = Semi-Non-negative Matrix Factorization. Factor M ≈ A · B where A is unconstrained, B is element-wise non-negative. Compromise between SVD (no constraints) and NMF (both non-negative). Used in CC critique because trained W_in is empirically non-negative (natural pre-ReLU sign pattern), motivating SNMF over plain SVD. Computed iteratively (alternating optimization), no closed form. Standard ref: Ding, Li, Jordan 2010.

## Strategic framing for the work test
Don't just re-run the existing demonstrations Stefan already has. Frame the task as:
> "How can I, in 2 hours, contribute one small piece of analysis that would be relevant to Stefan's open questions about M?"

Concrete shapes this could take:
- Vary M's properties (rank, condition, symmetry, sparsity) in a way the critique hasn't explored; report how the trained solution structure changes
- Probe a specific per-neuron-per-eigenvector relationship that goes beyond aggregate subspace claims
- Stress-test one of the existing findings with a new ablation (e.g., what if you use SNMF of M+cI for varying c?)
- Show that some "obvious" structural property of M predicts a feature of the trained model

A report that produces ONE such piece of new analysis — even partial — is what Stefan implicitly wants. Just confirming the critique is lower-value.

## Strategy
- **First 10 min/task**: read setup carefully. What's the metric? What's the baseline? What are candidate shortcuts? Also: what's the new analysis angle Stefan hasn't already done?
- **Run minimal experiments**: one plot showing "loss vs. shortcut magnitude" is often enough.
- **Don't soften conclusions.** If red-team finds shortcut → say so. If result survives → say "checked X, Y, Z and survives."
- **Use AI for code mods only** — not for designing experiments or interpreting results.

## TransformerLens quick reference
```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained('gpt2', device='mps')
tokens = model.to_tokens('some text')
logits, cache = model.run_with_cache(tokens)
# Hook names:
# blocks.{l}.hook_resid_pre, blocks.{l}.hook_resid_mid, blocks.{l}.hook_resid_post
# blocks.{l}.attn.hook_z, blocks.{l}.attn.hook_pattern
# blocks.{l}.mlp.hook_pre, blocks.{l}.mlp.hook_post
# Model dims: d_model=768, n_layers=12, d_mlp=3072, n_heads=12
```

## GPT-2 caveats and what's cached
**All four sizes cached and verified working on Peter's M2/24GB/MPS:**
| Model | Params | d_model | n_layers | d_mlp | Load (cached) | Fwd+cache |
|---|---|---|---|---|---|---|
| gpt2 | 163M | 768 | 12 | 3072 | 1.8s | 0.13s |
| gpt2-medium | 406M | 1024 | 24 | 4096 | 2.3s | 0.07s |
| gpt2-large | 838M | 1280 | 36 | 5120 | 2-3s | 0.35s |
| gpt2-xl | 1638M | 1600 | 48 | 6400 | 2-3s | 0.55s |

**Factual knowledge by size** (for prompt "The Eiffel Tower is located in the city of"):
- gpt2-small: top-5 is `the, a, London, its, danger` (no Paris)
- gpt2-medium: top-5 starts with `Paris, Lyon, Nice, E, T` (Paris #1)
- gpt2-large: top-5 starts with `Paris, E, Le, Vers, Lyon` (Paris #1)
- gpt2-xl: top-5 starts with `Paris, E, the, É, France` (Paris #1)

→ If task involves factual recall, **prefer gpt2-medium or larger.** gpt2-small is unreliable.
→ For probing studies, all sizes work — the activations exist regardless of completion quality.
→ For circuit analysis / interpretability that needs specific known circuits (induction heads, IOI), gpt2-small has the most published work and is usually the default.

## Papers in scope
- **APD** (arxiv 2501.14926v4, PDF at `~/Downloads/2501.14926v4.pdf`) — Braun, Bushnaq, Heimersheim, Mendel, Sharkey, Feb 2025
  - Decomposes weights into parameter components via faithfulness + minimality + simplicity (MDL framing)
  - Operates in **parameter space**, not activation space — components have shape of full θ
  - Minimality via attribution + top-k + sparse forward pass; attribution = ⟨∇_θ f, P_c⟩
  - 3 toy models tested: TMS (Elhage), Compressed Computation, Cross-Layer Distributed Reps
  - Claims to recover ground truth in all three
- **CC critique** — LW post by Heimersheim et al. (https://www.lesswrong.com/posts/ZxFchCFJFcgysYsT9/)
  - CC's gain comes from noise-fitting the mixing matrix M (= structure of W_E^T @ W_E)
  - When M=0 (clean labels), gain vanishes
  - Loss scales linearly with |M|
  - SNMF of M matches CC loss without any ReLU computation
  - Appendix finding: when x_42 = 1 only, noise at other outputs is **MLP-dominated** (not W_U W_E-dominated). Model **systematically undershoots** target y_42 = 2 because MSE forces trade-off between per-output fit and cross-output leakage. Consistent with noise-fitting picture.
- **Tension**: APD claimed CiS components in CC. If CC isn't CiS, what did APD find? Possibly the eigenvector structure of W_E^T @ W_E, dressed as computation-shaped objects.

## If task involves probing or memorization (approaches a / b from Stefan's writeup)
- **Probing**: linear probes can't distinguish "N atomic features" from "K features with N linear combinations of them." Need rank checks, intervention independence, behavioral dissociation.
- **Memorization**: MLPs-as-key-value-memories (Geva 2021); each fact ≈ key-value lookup. "Fact Finding" (Hernandez et al.) tried to localize facts → results were messy.

## What Claude should NOT do during the work task
- Write report prose
- Design experiments
- Interpret results / decide what's significant
- Make judgment calls about what to test

## What Claude SHOULD do
- Plot tweaks ("fix overlapping legends", "log-scale this axis")
- Code modifications ("loop over these 5 prompts instead of 1")
- API lookups ("what does cache.keys() return")
- Debugging ("this throws a shape error, here's the traceback")
- Quick sanity checks ("does this dimension calculation look right")
