# Three-Post Arc — Full Details

## Post 1: Why Softmax — READY TO PUBLISH
- File: `_posts/2026-02-17-why-softmax.md`
- **Summary**: Derives softmax as the solution to a constrained optimization problem: maximize expected query-key score subject to KL(π ∥ u) ≤ ρ. The KL budget ρ is framed as a hypothesis-testing "commitment to evidence" — if KL exceeds the budget, you have enough evidence to reject the uniform (ignorance) null. Relaxing the hard constraint yields the penalized form with inverse temperature β, recovering π ∝ exp(βz). The Interpretation section introduces ρ̂_eff = KL(π̂ ∥ u) = log n − H(π̂) as a diagnostic for trained heads, measuring effective commitment to evidence in a given context. The Further Implications section reintroduces β as a probe parameter and defines ∂ρ̂ = ∂_β ρ̂|_{β=1} = Var_π(z) as a temperature susceptibility, analogous to stat mech fluctuation-dissipation. The (ρ̂, ∂ρ̂) plane is proposed as a 2D diagnostic: monosemantic circuit heads should cluster at high ρ̂ / low ∂ρ̂ when activated; polysemantic heads at high ρ̂ / high ∂ρ̂.
- **References**: Cover & Thomas [^1] (Stein's lemma §11.8), Kardar [^4] (fluctuation-dissipation), Voita et al. [^5] (head pruning/confidence), Zhai et al. [^6] (entropy collapse)
- **Jaynes intentionally omitted** — his max-entropy derivation swaps objective/constraint vs. the blog's formulation; citing it was a red herring
- All pre-publish TODOs completed (spelling, notation legend, β motivation, log-growth claim, references, placeholder)

## Post 2: Attention Diagnostics — COMPLETE AND LIVE
- File: `_posts/2026-02-24-attention-diagnostics.md`
- See [post2-experiment-notes.md](post2-experiment-notes.md) for full experiment log
- See [blog-diagnostics.md](blog-diagnostics.md) for original theory/plan
- **Scratch notebook** (private, never publish): `notebooks/post2_attention-diagnostics/scratch/`
- **Curated notebook** (publish-ready): `notebooks/post2_attention-diagnostics/final/attention_diagnostics_peter.ipynb` — audited line-by-line, authorship note, savefig calls, executed end-to-end
- **Figures**: `notebooks/post2_attention-diagnostics/final/figs/` AND `assets/images/posts/attention-diagnostics/` (6 PNGs: fig1–fig6)
- **v3 prompts (maximally controlled)**: third name C, exact 15-token match, no duplicates
- **Key results**: |ΔKL| p=0.0002, |Δχ| p<0.0001 (circuit vs non-circuit). Both diagnostics clearly work.
- **Original hypothesis was WRONG**: signal is in the *shift* (ΔKL, Δχ) between activating/non-activating prompts, not in absolute position on the (KL, χ) plane
- **χ vs KL**: on a given prompt type, corr(KL, χ) = 0.34. But corr(ΔKL, Δχ) = 0.70 — shifts are correlated because both are query-key side statistics.
- **KL + Var_v > KL + χ for fingerprinting**: (KL, Var_v) is now the canonical 2D fingerprint; χ demoted to footnote/future work.
- **ΔKL direction**: selection heads INCREASE KL on IOI (more selective). Structural heads ≈ 0. S-Inhibition mixed.
- **Prompt control lesson**: 3 iterations of non-IOI prompts. Each improved control changed results dramatically.
- **Remaining confound**: layer depth (corr ≈ 0.38). Context length eliminated.
- **Key tension with Post 1**: Post 1 predicted χ should be low in both conditions (Δχ ≈ 0), but data shows significant Δχ (p<0.0001). Acknowledged in limitations; deferred to Post 3.
- **Post 3 will open with**: χ wasn't good enough — both KL and χ are query-key side. Var_v adds the value axis.

## Post 3: Feature Upgrade + First-Pass Circuit Graph (planned)
- See [post3-plan.md](post3-plan.md) for full plan from Mar 2026 conversation
- **Experiments needed first**: validate ΔVar_v discriminates circuit vs non-circuit heads; corr(ΔKL, ΔVar_v) should be << corr(ΔKL, Δχ) ≈ 0.70
- **Three sections**: (1) motivate Var_v over χ with data; (2) factor analysis on log(Var_v) at last token; (3) J_diff circuit graph
- Feature vector: log(Var_v) per attention head + Var_W_out per MLP, all at **last token position only**
- J_diff = W_act Wᵀ_act − W_null Wᵀ_null (Gaussian hidden units / factor analysis for now)
- J_diff cancels layer-depth confound and structural MLP-attention correlations

## Post 4: Binary RBM + MI Regularization for Mode Selection (planned)
- Upgrade from Gaussian to binary hidden units (circuits are on/off — binary more natural than Gaussian)
- Kyle's factored-MPF code + MI regularization between visible/hidden spins: selects most significant modes first, solves rotational ambiguity
- Kyle likely co-author; Bialek academic lineage (Peter + Kyle → Stephanie Palmer → Bill Bialek → same lineage as Dario Amodei)
- Training dynamics / Kim 2026 / phase transitions deferred to Post 5
