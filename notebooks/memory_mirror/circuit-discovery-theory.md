# Circuit Discovery Theory (Mar 2026)

## The Circularity Problem

The fingerprinting approach (ΔKL, Δout_mag) requires defining activating vs. non-activating prompts — but that requires knowing what the circuit does. For IOI this was fine because Wang et al. gave the answer key. For novel circuits (e.g., chess), you don't have ground truth.

**Workarounds**:
- **Behavioral proxy**: Use model's output as activation signal. Positions where Opus correctly identifies a tactic = activating; matched non-tactical positions = control. No prior knowledge of circuit needed.
- **Contrastive minimal pairs**: Near-identical positions differing by one piece, changing the correct move. Clean signal, hard to generate at scale.
- **Unsupervised first**: Run many prompts, compute fingerprints for all heads, cluster — then interpret clusters afterward.
- **Fingerprint → ablation → confirm**: Diagnostics nominate candidates cheaply; activation patching confirms causality expensively.

## Three Circuit-Structure Hypotheses

| Hypothesis | Structure | Per-head Δ profile | Ablation effect |
|---|---|---|---|
| Sparse circuit | Few heads, each necessary | Few heads, large Δ | Ablating one head hurts a lot |
| Voting/lookup | Many heads, each sufficient | Many heads, small uniform Δ | Ablating one barely matters |
| Reservoir | Unstructured distributed | High Var_v, low task-specific Δ | Graceful degradation ∝ fraction ablated |

ICA addresses a fourth case: **structured distributed computation** — multiple heads jointly implement sub-circuits, overlapping across sub-circuits. ICA finds independent activation patterns across heads (sub-circuits as latent sources), not residual stream features. This is a computational claim, not representational.

## Superposition Clarification

**Superposition is a representational claim** (features stored in residual stream as near-orthogonal directions, more features than dimensions). It is NOT a circuit-structure hypothesis. Don't conflate. The three hypotheses above are about circuit structure independent of whether representations are superposed.

ICA in this project is NOT about recovering residual stream features — it's about finding independent sub-circuit activation patterns across heads.

## Transcendence Paper

**Zhang et al. (Edwin Zhang), NeurIPS 2024**: "Transcendence: Generative Models Can Outperform The Experts That Train Them"
- URL: https://arxiv.org/abs/2406.11741
- Chess LLMs trained on games from weak players can exceed those players' ELO via majority voting
- Mechanism: ensemble of weak players — more of them avoid clear blunders than make them
- Low-temperature sampling is the key lever (amplifies consensus signal)
- Implication: chess LLMs may not implement tactical reasoning but distributed voting/lookup
- Circuit discovery on chess would find "position similarity" or "aggregation" heads, not "tactic computation" heads

## Null Result for Chess

If voting is the mechanism:
- ΔKL should be roughly flat across tactic types (fork vs. pin vs. positional)
- You'd see a "chess vs. not-chess" split but no finer tactic-type structure
- Many heads activate broadly on chess positions, none specialized by tactic

If sparse circuits exist:
- Different tactic types recruit partially distinct head sets
- Fingerprints on fork vs. non-tactical and pin vs. non-tactical diverge

**The test**: generate three groups — fork positions, pin positions, non-tactical positions. If Δ profiles for (fork vs. non-tactical) and (pin vs. non-tactical) are the same, that's voting. If they diverge, that's circuit structure.

## Agentic Loop Idea

Claude as the "scientist" driving prompt generation:
1. Generate diverse chess positions, run fingerprints, cluster heads
2. Analyze clusters — what do they seem to track?
3. Design more targeted prompts (specific tactic types, minimal pairs) to test hypotheses
4. Track Δ and covariance between/within prompt groups
5. Iterate until clusters stabilize or hypothesis falsified

**Key requirement before building**: define what a null result looks like (above) so the agent tries to falsify, not just find clusters.

**Instructions get simpler as methodology matures**: once you have a reliable discriminator (like contrastive ICA), the agent just needs: "find heads with large Δ on this task, design prompts to isolate what property they're tracking."

## Reservoir Hypothesis

Under reservoir computing:
- Many heads contribute small, distributed activations; high redundancy
- No individual head necessary — graceful degradation under ablation
- Fingerprint shows many heads with moderate diffuse ΔKL
- Var_v broadly high (rich dynamics) but not task-selective
- Ablation of random subsets: loss roughly proportional to fraction ablated, not which ones

Voting and reservoir both look "diffuse" in per-head fingerprints — ablation signature discriminates them.

## These hypotheses may coexist across tasks
- Chess: likely voting/lookup (Transcendence)
- IOI: likely sparse circuits (Wang et al.)
- General language: likely reservoir
That taxonomy itself would be a useful result.

## Scalable Pipeline: CLT + Vectorized ICA (Mar 2026)

Current circuit discovery (ACDC, path patching, attribution patching) requires:
- Human-defined task + metric upfront
- Human-designed clean/corrupted prompt pairs
- One circuit at a time, doesn't transfer

**The scalable alternative**:

1. **CLT decomposition** — compute cross-layer transcoder features once per model. Gives interpretable basis for residual stream at each layer. Anthropic building these at scale (Biology of LLMs). Inherited, not recomputed.

2. **Project x into CLT feature basis** — x_feat ∈ ℝ^{T × n_features}, sparse (most features inactive per prompt). Compresses x from T×d_model to T×n_features where n_features << d_model for active features.

3. **Vectorize** — x_feat_vec ∈ ℝ^{T·n_features}. Manageable because sparse. This dissolves the "which token" problem for ICA — joint position×feature space.

4. **Sparse ICA on x_feat_vec** — unsupervised, run on large diverse corpus. ICs are independent activation patterns over (position, named-feature) pairs. No task-specific prompts needed. Human labels components *after* the fact.

5. **Capacity ranking for cross-validation** — rank all heads by C(W_QK) from singular values. Parameter-only, one-time, instant. High-capacity heads should correspond to ICA components with large variance. Connects static (parameter) and dynamic (activation) views without intervention.

**Why this scales**:
- CLT features: one-time per model, reused for every circuit
- ICA: unsupervised, parallelizable, scales with data not human effort
- Capacity: O(n_heads × d_model²), no inference needed
- Human bottleneck pushed to end, applied only to top components

**Why ICs are interpretable here** (unlike raw ICA on x):
- Features are CLT features = already labeled (e.g. "the 'John' name feature")
- ICs become: "independent activation pattern over (position, CLT feature)" — readable as circuit components
- A circuit is a set of correlated ICs that jointly implement a computation

**Potential framing**: "unsupervised circuit discovery via sparse ICA on CLT-projected residual streams." IOI circuit = proof-of-concept validation. Methodology paper independent of any specific finding.

**Connection to 4-way head characterization**:
- ICA on x_feat_vec → OV/consequence side (what gets written, in interpretable coordinates)
- Capacity + W_QK singular vectors → QK/mechanism side (what triggers the head, in same coordinates)
- Full circuit: "head H reads CLT feature F_q at position i, writes CLT feature F_v at position j"
- Both sides expressed in same CLT feature basis → unified description
