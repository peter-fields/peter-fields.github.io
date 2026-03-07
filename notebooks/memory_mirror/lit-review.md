# Literature Review — Feb 2026

Full novelty assessment for the attention diagnostics work (Posts 1-2).

## 1. Expanding the IOI Circuit / Finding Additional Heads

**Nobody has done this.** Papers that cite Wang et al. 2022:
- Merullo et al. 2024 (ICLR) — tested circuit reuse across tasks in GPT-2 Medium, didn't find new heads
- Nainani et al. 2024 — tested circuit generalization across prompt formats, didn't find new heads
- Conmy et al. 2023 (ACDC, NeurIPS) — automated recovery *missed* some Wang et al. heads, didn't find new ones
- Wang et al. "Lessons Learned" post on Alignment Forum flagged open questions about S-Inhibition but no follow-up published

**Our L8H1/L8H11 finding is novel** — no one has reported additional functional heads in the IOI circuit.

## 2. KL from Uniform as Per-Prompt Circuit Diagnostic

**Known**: attention entropy as a metric (Voita 2019 for pruning, Zhai 2023 for training stability, Zheng et al. 2024 survey).
**Novel**: using KL(π ∥ u) per head per prompt as a circuit diagnostic. Nobody uses ΔKL between conditions.

Related:
- Voita et al. 2019 — "head confidence" (avg max attention weight), used for pruning not discovery
- Zhai et al. 2023 — per-head entropy during training, not inference-time circuit discovery
- Bali et al. 2026 (arXiv:2602.16740) — attention head stability across training runs
- "Entropy Meets Importance" 2025 (arXiv:2510.13832) — combines entropy + importance for pruning
- Draye et al. 2025 (arXiv:2512.05865) — sparsity regularization for interp, different approach

## 3. Temperature Susceptibility / Fluctuation-Dissipation

**Closest competitor: Kim 2026** (arXiv:2602.08216, published Feb 9 2026)
- Derives softmax from free energy minimization (Helmholtz)
- Defines "effective specific heat" from attention energy fluctuations — same math as our χ
- Uses it to detect grokking as thermodynamic phase transition during *training*
- Does NOT: propose (KL, χ) phase space, test against labeled circuits, use for inference-time head characterization, use contrastive prompts

**Our differentiation**: inference-time head typing, 2D diagnostic plane, contrastive prompt methodology, circuit validation.

Other:
- Nick Ryan 2024 — learnable per-head temperature, does not compute derivative or relate to susceptibility
- Masarczyk et al. 2025 — temperature effects on representations, output softmax focus
- Geshkovski et al. 2024 (arXiv:2410.06833) — metastability in self-attention, theoretical
- Tiberi et al. 2024 (NeurIPS, arXiv:2405.15926) — stat mech theory of multi-head attention, theoretical
- Entropic OT 2025 (arXiv:2508.08369) — softmax as KL-regularized OT, no empirical diagnostics

## 4. Cross-Head Correlation / Functional Connectivity

**Nobody has done this.** The "fMRI functional connectivity" analogy for attention heads does not exist in the literature.

Related but different:
- Elhelo & Geva 2025 (MAPS) — infers head functionality from weight matrices, not runtime correlations
- robertzk 2024 (Alignment Forum) — inspected all GPT-2-small heads with SAEs, feature-based not correlation-based
- Nam et al. 2025 (NeurIPS, arXiv:2505.13737) — causal head gating, learns soft gates, captures interactions through joint optimization but not pairwise statistical correlations
- Anthropic 2025 Circuit Tracing — attribution graphs, not statistical correlations of summary metrics

## 5. Observational (Non-Causal) Circuit Discovery

**Our approach is the only one that proposes: run model on two prompt sets, measure distributional statistics, compare, identify circuit heads from differences.**

Closest:
- CD-T (Hsu et al. 2024, ICLR 2025, arXiv:2407.00886) — analytical decomposition, no intervention, but requires mathematical forward-pass decomposition, not purely statistical
- MAPS (Elhelo & Geva 2025) — parameter-based, no inference needed at all, but different methodology entirely
- EAP/Attribution Patching (Syed et al. 2024) — gradient approximation of patching, still requires clean/corrupted pairs and gradients
- ACDC (Conmy et al. 2023) — full causal, expensive

## Full Citation List (with URLs)
- Zhai et al. 2023 — attention entropy collapse during training (https://arxiv.org/abs/2303.06296)
- Nick Ryan 2024 — learnable per-head temperature (https://nickcdryan.com/2024/08/02/introducing-a-learnable-temperature-value-into-the-self-attention-scores/)
- Masarczyk et al. 2025 (Unpacking Softmax) — temperature effects on representations (https://arxiv.org/html/2506.01562v1)
- Wang et al. 2022 — IOI circuit, head labels (https://arxiv.org/abs/2211.00593)
- IAAR-Shanghai / Zheng et al. 2024 — Awesome Attention Heads survey (https://arxiv.org/abs/2409.03752)
- Voita et al. 2019 — head pruning, confidence metric (ACL 2019, https://arxiv.org/abs/1905.09418)
- Kardar 2007 — Statistical Physics of Particles, Ch. 4 (fluctuation-dissipation)
- **Kim 2026** — "Thermodynamic Isomorphism of Transformers" (https://arxiv.org/abs/2602.08216). Closest competitor. MUST CITE.
- Entropic OT 2025 — "Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport" (https://arxiv.org/abs/2508.08369)
- Conmy et al. 2023 (ACDC) — automated circuit discovery via activation patching (https://arxiv.org/abs/2304.14997)
- Merullo et al. 2024 — circuit component reuse across tasks (https://arxiv.org/abs/2310.08744)
- Elhelo & Geva 2025 (MAPS) — infer head functionality from parameters alone (https://arxiv.org/abs/2412.11965)
- Hsu et al. 2024 (CD-T) — analytical circuit discovery without intervention (https://arxiv.org/abs/2407.00886)
- Syed et al. 2024 (EAP / Attribution Patching) — gradient approximation of activation patching (https://aclanthology.org/2024.blackboxnlp-1.25.pdf)

## TODO: Further Lit Searches
- [ ] Search for "functional connectivity" + "transformer" or "neural network" — neuroscience-inspired approaches
- [ ] Check if anyone has applied spectral clustering (not just hierarchical) to attention head statistics
- [ ] Search for perturbation-based methods in neuroscience applied to transformers (virtual lesion studies → activation patching is already this, but statistical analogs?)
- [ ] Look for any work using covariance structure of model internals across inputs as a diagnostic (not just attention — could be MLP activations, residual stream)
- [ ] Check if Kim 2026 cites or is cited by anything relevant (it's very new)
- [ ] Search for "attention head taxonomy" or "head classification" approaches beyond the ones found
- [ ] Look for prior work on the specific entropic OT derivation vs. our KL-constrained formulation — are they equivalent?
