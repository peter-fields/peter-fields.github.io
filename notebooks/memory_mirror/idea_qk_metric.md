---
name: idea_qk_metric
description: Research idea — W_QK = G + B decomposition; shared content-matching metric and directed-routing 2-form across heads; content/compute subspace split
type: project
---

## W_QK = G (metric) + B (2-form): Shared Geometry of Attention Heads

**Core observation**: W_QK^h = G^h + B^h where:
- G^h = (W_QK^h + W_QK^h^T) / 2 — symmetric bilinear form (proto-metric, content matching)
- B^h = (W_QK^h - W_QK^h^T) / 2 — antisymmetric 2-form (directed routing, generalized geometry B-field)

Attention score = x_q^T G^h x_k + x_q^T B^h x_k = symmetric content similarity + directed routing preference.

**The shared structure claim**: G_crude = mean_h G^h and B_crude = mean_h B^h capture what all heads collectively agree on — the shared geometry of the communication bus.

**Mathematical framing**: W_QK is a general bilinear form = metric + 2-form. This is exactly the structure of Generalized Geometry (Hitchin): g + B-field. The B-field creates torsion — the reason K-composition and Q-composition are geometrically distinct operations.

**Connection to Elhage "subspaces" — why the story is ill-defined without G:**
- Head A writes output^A ∈ V* (row vector added to residual stream). Head B reads via rows of W_K^B (also in V*).
- "Head A's write lands in the subspace head B reads from" = row space of W_O^A overlaps with column space of W_K^B in R^{d_model}.
- But measuring overlap requires an inner product on V*. Without one, the angle between any two directions in V* is coordinate-dependent. Euclidean is one arbitrary choice.
- Each head provides its own canonical pairing V* × (head space) → R — d_head linear functionals on V*. Different across heads, no canonical way to combine them into a metric on V*.
- G^{μν} is the ONLY inner product on V* that appears in the actual computation (attention score = x_q G^{μν} x_k). With G, "head A writes v, head B reads it" means: x_q G v = content-matching contribution of v to head B's attention. Precise and coordinate-free.
- **Blog point**: Elhage's subspace story is literally ill-defined by his own no-privileged-basis argument. G is the missing piece that makes it well-defined.

**What G should span (key conceptual point):**
G is the metric of the *residual stream as a whole* — it must span BOTH content and compute directions. A head needs to (a) attend to the right tokens via W_QK (uses G), (b) read their information via W_V (uses G), and (c) write into directions later heads can use via W_O (also uses G, but now those directions may be compute-space). So G can't be purely W_E space. The ~0.69 W_E mass for G top modes means G is mostly content-aligned at early layers, but the remaining ~0.31 is compute directions — that's the channel for head-to-head communication via processed representations, not noise.

**Why B is W_QK only:**
B creates asymmetry in "who attends to whom" — a routing preference independent of content. W_OV has no equivalent: it's a linear read-write map, not a routing selector. There's no meaningful antisymmetric part of W_OV to extract. B belongs to the attention mechanism; G belongs to the residual stream.

**The shared G/B is the big claim:**
- Shared G = the common geometric language the residual stream speaks. Every head reads from and writes to the same G-metric substrate. Without shared G, K-composition would be coincidental — head A's write directions would have no reason to align with head B's read directions.
- Shared B = the common routing grammar (prev-token, skip positions, BOS sink, etc.) all heads participate in, without independently rediscovering it.
- Together: G tells you the geometry of the residual stream; B tells you the directed routing structure layered on top of it.

**Claim framing (loosened from original)**: Not "this is the geometry of language" but "an interesting coordinate-aware decomposition of what all heads collectively compute — potentially a step toward coordinate-free interpretability."

---

## Empirical Results (2026-03-18, GPT-2 small + attn-only-2l)

Scripts: `notebooks/post4_qk_metric/scratch/run_exp1_G_spectrum.py` through `run_exp4_content_vs_compute.py`

### Key finding: B top modes live in the compute subspace

W_E projection mass (90% variance subspace of W_E):
- G eigenvectors: mean ~0.69 (mostly in token-identity / W_E space) — confirmed
- B top singular vectors: mass ~0.003–0.07 for top-4 modes → essentially OUTSIDE W_E space
- B lower modes: converge back toward W_E space (~0.4–0.6)

**attn-only-2l** (cleanest model — no MLPs, K-composition is everything):
- B SV1, SV2 (σ=1.95): W_E mass = 0.003, 0.002 — pure compute subspace
- B SV3, SV4 (σ=1.54): W_E mass = 0.026, 0.056 — still nearly pure compute
- These are the K-composition channels (Previous Token → Induction)

**GPT-2 small**:
- B SV1, SV2 (σ=1.00): W_E mass = 0.067, 0.053 — mostly compute
- B SV3, SV4 (σ=0.98): W_E mass = 0.284, 0.154 — moderate

### Other findings
- B_crude Frobenius norm > G_crude norm (especially attn-only-2l: 4.16 vs 2.56) — directed routing is the dominant shared signal
- B singular values come in exact pairs (skew-symmetric signature confirmed)
- attn-only-2l B_crude captures 53% of per-head B norm; GPT-2 only 28% — tighter coordination in simpler model
- G_crude / mean head G norm ≈ 0.26 for GPT-2 — most QK structure is head-specific, not shared
- G_crude is indefinite (319 pos, 449 neg eigenvalues for GPT-2) — not a Riemannian metric, closer to pseudo-Riemannian / Lorentzian

### Content vs compute split
- **G** = content matching, lives in W_E subspace (~0.69 W_E mass)
- **B top modes** = directed compute routing, lives OUTSIDE W_E subspace (~0.003–0.07 mass)
- This is the formalization of "compute subspace" — not orthogonal to W_E in the naive sense, but accessible via B's top singular modes

### Vocab projections (filtered, top-1% norm tokens removed)
**GPT-2 small G_crude top eigenvectors**:
- EV1 (λ=0.705): closing punctuation/quotes vs discourse connectors
- EV3 (λ=0.488): proper nouns/events vs formatting
- EV6 (λ=0.382): persistence/heritage words vs file/tech terms

**GPT-2 small B_crude top modes**:
- SV1 (σ=1.00): query=common tokens [the, ", (, a] → key=punctuation/separators [, - . ...]
- SV3 (σ=0.98): query=common tokens → key=rare/obscure tokens (and vice versa)
- Pattern: B routes from high-frequency query contexts toward structurally complementary keys

---

## Exp5 Results (2026-03-18, attn-only-2l)

Scripts: `notebooks/post4_qk_metric/scratch/run_exp5_shared_GB.py`, `run_exp5_geometric_foundation.py`

### Test: B from W_QK predicts K-composition in W_OV
Project both W_O write directions and W_K read directions onto B_crude's RIGHT singular vectors (key-side of B). Correlation with raw K-comp alignment:
- **B-filtered: 0.678**
- **G-filtered: 0.174**

B predicts K-composition structure significantly better than G. G-filtered heatmap is essentially flat (all pairs ~0.27–0.38) — G does not discriminate head pairs. B preserves the sparse structure: L0H0→L1H6, L0H2→L1H6 stand out in B-filtered, matching raw signal (L0H0→L1H6=0.091, L0H3→L1H6=0.100). Anomaly: L0H4 dominates B-filtered (~0.38) without a strong raw alignment signal — unexplained.

**Implication**: B was extracted from W_QK routing geometry alone. It also predicts W_OV communication structure. This is evidence that B is a shared object, not just a W_QK decomposition artifact.

### Test: Stacked SVD per matrix type (rotation-invariant across heads)
Stack all heads of each matrix type → SVD → shared directions in d_model. W_E projection mass:
- **W_V: mean=0.856** — almost entirely content/token-identity space. Cleanest single result.
- W_Q: mean=0.402 — mixed; top-5 W_E mass [0.006, 0.09, 0.016, 0.071, 0.690] — top modes are compute
- W_K: mean=0.485 — mixed; top-5 [0.083, 0.002, 0.002, 0.025, 0.330] — top modes are compute
- W_O: mean=0.384 — mixed

Principal angle cosines between stacked subspaces (top-32, random baseline ≈0.25):
- **W_Q ↔ W_K: 0.428** — well above random. Genuine shared subspace = G.
- W_K ↔ W_O: 0.290 — borderline (K-composition channel?)
- W_Q ↔ W_V: 0.164, W_K ↔ W_V: 0.174, W_V ↔ W_O: 0.105 — at or below random

**Key findings**:
1. W_V is pure content — reads almost entirely from W_E subspace
2. W_Q ↔ W_K share a genuine subspace (G) — routing uses the same directions
3. W_Q and W_K's TOP (highest sv) shared directions are COMPUTE-type (low W_E mass) — the most concentrated routing signal is outside token space
4. W_V and W_O barely share directions — the read/write content channel is not simple

### What does NOT work (naive approaches)
- Per-head G consistency: principal angles between G_QK^h, G_V^h, G_O^h within each head are near-random (~0.10–0.17, baseline ~0.125). Simple averaging across heads does not recover a consistent G from all three sources.
- Pooled SVD of all weight directions: cannot separate G (symmetric) from B (antisymmetric) — wrong tool. G and B require the bilinear form structure of W_QK.

### Open next steps for exp5
- Fix L0H4 anomaly in B K-comp heatmap
- Run stacked SVD analysis on GPT-2 for comparison (does W_V stay pure content in larger model?)
- Deeper look at W_Q/W_K shared compute directions (top modes, low W_E mass) — are these B's singular vectors?
- Cross-seed stability test on attn-only-2l (train multiple seeds, compare G eigenspaces)

---

## SAE / CLT Comparison (TODO — future work)

**The hypothesis**: CLT features (cross-layer transcoder dictionary) are trained on residual stream activations dominated by W_E structure. They are likely blind to B's top compute-subspace modes.

**Predicted finding**: G eigenvectors should align with CLT/SAE features (both in W_E space). B's top modes should NOT align with any CLT features — they are the directions CLT attribution graphs systematically miss.

**Setup needed**: sae_lens requires Python 3.10+. conda base is Python 3.9.
- **Do NOT upgrade base env** — would break TransformerLens, scipy, scikit-learn builds
- **Create new env**: `conda create -n py311 python=3.11 && pip install transformer-lens sae-lens torch numpy matplotlib scipy scikit-learn`
- Joseph Bloom's GPT-2 small SAEs on HuggingFace: `jbloom/GPT2-Small-SAEs-Reformatted`
- Compare: cosine similarity between G eigenvectors / B singular vectors and SAE decoder columns

---

## Theoretical Framework — Key Claims (2026-03-18)

### Two-post arc
- **Post 1** (weight-based, runnable now): Extract G and B jointly from W_QK symmetric/antisymmetric parts AND W_OV (both W_V row space = read-for-content, and W_O column space = write). Compare against W_E, W_U. Show implicit-G critique of current Frobenius/cosine alignment analyses. Model: attn-only-2l primarily.
- **Post 2** (requires GPU training): Train attn-only-2l on multiple seeds. Cross-seed Procrustes alignment of G_crude. Principal angles between top-k G eigenspaces across seeds. Near-zero → territory; scattered → map.

### Low rank per head vs full rank shared G

Per head: W_QK^h is rank ≤ d_head = 64 (in d_model = 512 space). Each head projects through a 64-dimensional bottleneck — it only "sees" a d_head-dimensional slice of the residual stream geometry. Different heads CAN use completely orthogonal subspaces of d_model (Elhage: low rank per head enables specialization). W_Q and W_K each project d_model → subspace of d_model via the d_head bottleneck. The head space (R^{d_head}) is a computational artifact, not a geometrically meaningful object. What's real is the projection W_QK = W_Q W_K^T as a rank-d_head operator within d_model.

Shared G: probably rank d_model (or high rank) — the full geometry of the residual stream. Each W_QK^h sends most of G's directions to zero (only sees its own d_head-dimensional slice). G_crude = mean_h G^h is the aggregate: high-eigenvalue directions are "highways" used by many heads collectively; low-eigenvalue directions are "side streets" used by fewer or weakly.

Sloppiness of G_crude reframed: not a property of G's intrinsic structure but of how heads collectively sample G. The non-flat spectrum says heads don't fully orthogonalize — some directions are genuinely shared across multiple heads. If heads fully tiled d_model with orthogonal subspaces, G_crude would be flat. Non-flat = shared structure exists.

### Sloppy G and softmax
- G's eigenvalue spectrum is **log-spaced** (sloppy), spanning many orders of magnitude
- This gives G **high information capacity** — each decade of eigenvalues encodes similarity at a different linguistic scale (morphological, syntactic, semantic, discourse)
- Low eigenvalue modes still contribute at their appropriate scale — not discarded
- Softmax is the natural matched readout: it operates on log-scale differences (exp(score)/Z), so it integrates the full log-hierarchy while remaining invariant to overall scale
- Softmax temperature (1/sqrt(d_head)) sets where in the log-hierarchy the decision boundary sits
- **If G is sloppy and stable across seeds**: the model is being forced by language data to preserve a pre-existing multi-scale linguistic hierarchy → strong territory argument

### Scale/gauge freedom
- G from W_QK is scale-underdetermined: softmax invariance means G is a projective object (directions matter, not magnitudes)
- W_OV has no equivalent invariance — its singular values set absolute scale → W_OV provides the scale anchor for joint extraction
- Fix for joint extraction: normalize each W_QK^h by spectral norm before averaging; let W_OV singular magnitudes set scale
- LayerNorm in larger models likely fixes the gauge automatically by anchoring residual stream norm

### Rotation/gauge freedom in aggregation
- Averaging W_QK^h directly assumes all heads share the same coordinate frame — not true
- Right approach: rotation-invariant aggregation (Grassmannian averaging of per-head eigenspaces)
- Current G_crude still gives valid content/compute split (B W_E mass result is robust) but "shared geometry" claim needs the rotation-invariant version

### Full read/write comparison for Post 1
| Matrix | Side | Role |
|--------|------|------|
| W_Q, W_K | read/routing | G_QK — who attends to whom |
| W_V | read/content | what gets selected when attended |
| W_O | write | what gets deposited into residual stream |
| W_E | reference | token identity input |
| W_U | reference | token identity output |

G should be consistent across all five if it's the shared residual stream geometry.

### W_OV convention (Peter's, = TransformerLens)
W_OV = W_V @ W_O (row vector convention). In SVD(W_V @ W_O) = U S Vh:
- **U columns** = column space of W_V = **read directions** from residual stream
- **Vh rows** = row space of W_O = **write directions** into residual stream

W_OV does NOT decompose as G + B (additive). It is already factored as a product W_V @ W_O.
For induction-type heads: W_V reads from G-space (content), W_O writes into B-space (compute) — so W_OV is a G*B bridge (inter-subspace coupling). The symmetric/antisymmetric decomposition of W_OV as a linear map does not cleanly recover this structure.

### 4-number head profile (exp5 target)
Both W_V and W_O empirically read/write from mixtures of content and compute. Don't assume prior structure — measure it. Per head:
1. W_QK symmetric part W_E projection mass → G^{μν} content-matching fraction
2. W_QK antisymmetric part W_E projection mass → B^{μν} compute-routing fraction
3. U columns W_E mass (column space of W_V) → how much head reads from content
4. Vh rows W_E mass (row space of W_O) → how much head writes into content

**2D scatter for exp5**: x = W_V read W_E mass, y = W_O write W_E mass, one point per head.
- Top-right: content-copy heads
- Bottom-left: compute-to-compute heads
- Top-left: induction-type (reads content, writes compute)
- Bottom-right: reads compute, writes content (?)
For attn-only-2l (16 heads total) every point can be labeled.

### Implicit-G critique (punchy blog point)
- Current circuit alignment analyses (e.g. "W_O of head A aligns with W_K of head B") use Euclidean/cosine similarity = implicit identity metric
- Correct measure: W_O^A^T G W_K^B (G-weighted inner product)
- Making G explicit can change what aligns with what — demonstrable with a concrete example

---

## Joint Learning of G and B — Deferred (complex, do later)

**Core idea**: G and B are latent coordinate-free objects. Every W_QK^h and W_V^h (and W_O^h) is a head-specific projection of the same underlying geometry. Learn G and B jointly from all constraints rather than computing separate estimates and checking consistency.

**Why naive mean fails**: pointwise mean_h sym(W_QK^h) is not rotation-invariant. Two heads implementing the same geometry in different coordinate frames partially cancel. Same problem for mean_h (W_V^h @ W_V^h^T).

**Rotation-invariant formulation**: work on the Grassmannian. For each head h compute projection matrices (coordinate-free subspace objects):
- P_h^QK = projection onto top-k eigenvectors of sym(W_QK^h)
- P_h^V = projection onto top-k eigenvectors of W_V^h @ W_V^h^T
- P_h^O = projection onto top-k eigenvectors of W_O^h^T @ W_O^h

Form pooled projection matrix:
C = w_QK * mean_h P_h^QK + w_V * mean_h P_h^V + w_O * mean_h P_h^O

**G eigenspace** = top-k eigenvectors of C. Closed form, rotation-invariant by construction.

**Scale**: G eigenspace is gauge-free from W_QK alone (softmax invariance). W_V^h @ W_V^h^T has no gauge freedom — its eigenvalues have physical units (squared weight norms). w_V anchors the scale. Natural: normalize weights so each source contributes equally.

**For B**: same structure using P_h^B = projection onto top singular vector pairs of anti(W_QK^h). C_B = mean_h P_h^B. B top singular vectors = top eigenvectors of C_B.

**W_O slot**: W_O rows are covectors — (0,2) slot, dual to G^{μν}. Two options:
- Include P_h^O directly in C (valid if G is roughly isotropic within content subspace)
- Use as post-hoc consistency check on dual slot

**Free parameters**: k (rank of G — read off eigenvalue elbow of C), w_QK / w_V / w_O (relative source weights — start equal, normalized by head count).

**Literature**: this is multi-view subspace learning. See Shared Response Model (Chen et al. 2015), multi-view CCA. May have useful algorithmic tricks.

**Why this matters**: G learned this way is the coordinate-free substrate that all W_OV and W_QK jointly learned to act upon. Not an artifact of any single head's coordinate frame. If stable across seeds → territory (geometry forced by language data). If unstable → map.

**Connection to sloppy structure**: k should be set by the eigenvalue elbow of C. Sloppy C (log-spaced eigenvalues) → many scales of geometric structure encoded in G → high information capacity.

---

## Exp6 Results (2026-03-18, attn-only-2l) — Basis Change

Script: `notebooks/post4_qk_metric/scratch/run_exp6_basis_change.py`

**Key insight**: G+B decomposition is forced (exact algebra). The only question is what basis makes things interpretable. Pick basis from heads with known jobs:
- G basis: eigenvectors of G^{induction} (L1H5, max ||G||_F in L1)
- B basis: singular vector pairs of B^{prev-token} (L0H5, max ||B||_F in L0)

Coefficient table reveals head types immediately:
- L0H5 (prev-token): G_total=0.278, B_total=10.895 — pure routing
- L1H5 (induction): G_total=32.949, B_total=0.671 — pure content matching
- L1H6, L1H7: G_total≈0.5, WV_G≈5, WO_G≈10 — heavy W_V/W_O on G basis, content-copy type?

**Core blog claim**: The basis change doesn't affect the model computation at all. It's just coordinates. The fact that a single natural coordinate choice (from known head types) makes all other heads legible is the interpretability result.

---

## Exp6b Results (2026-03-18) — Vocab Labels + Sloppy Spectrum

Script: `notebooks/post4_qk_metric/scratch/run_exp6b_vocab_labels.py`

- G dir 12: pure BE-verb tense ("been/being/were/be") — a clean semantic direction
- G spectrum is sloppy (log-spaced eigenvalues ≈ 2.85, 2.36, 2.26, 2.23...) — multi-scale content geometry
- B basis (L0H5) singular vectors come in pairs (σ=1.599, 1.599, 1.326, 1.326...) — canonical skew-sym form

---

## Exp8 Results (2026-03-18) — Sloppiness Across All Heads

Script: `notebooks/post4_qk_metric/scratch/run_exp8_sloppiness.py`

Sloppiness metrics (G_slop = log10(|λ_1|/|λ_K|), G_PR = participation ratio):
- **L0H4 is anomalous**: G_slop=1.020, G_top1=23.5%, B_slop=0.883 — dominant single mode in both G and B. Concentrated, specific job. Matches the unexplained L0H4 anomaly in exp5 B K-comp heatmap.
- **L1 induction heads** (L1H0–L1H5): G_slop ≈ 0.17–0.30, G_PR ≈ 15.7 — near-uniform G spectrum. Must match ANY token → needs all content directions equally.
- **L0H5 (prev-token)**: B_slop=0.149, B_PR=15.8 — FLAT B spectrum. Large ||B||_F but spread uniformly. Routing is positional (not token-identity), so B looks flat in token-content space. B spectral flatness = routing ignores token identity.

**Sloppy G diagnostic**: sloppy G → multi-scale content matching. Flat B → routing by position not content.

---

## Exp9 Results (2026-03-18) — G-corrected K-composition

Script: `notebooks/post4_qk_metric/scratch/run_exp9_G_corrected_kcomp.py`

Elhage K-comp (actual formula): ||W_QK^{L1} W_OV^{L0}||_F / (||W_QK^{L1}||_F ||W_OV^{L0}||_F) where W_QK = W_Q W_K^T and W_OV = W_V W_O (both d_model × d_model). Matrix cosine similarity. Still Euclidean/Frobenius — basis-dependent.
Exp9 computed factored approximation: ||W_K^T G W_O^T||_F / sqrt(tr(W_O G W_O^T) · tr(W_K^T G W_K)) — uses component matrices, not full virtual weights. The full G-corrected version would be ||W_QK G W_OV||_F / (||W_QK||_G ||W_OV||_G).

**Correlation between standard and G-corrected: 0.323** — they disagree substantially.

Standard: sparsely highlights L0H0→L1H6 (0.091) and L0H3→L1H6 (0.100) as induction circuit.
G-corrected: completely different picture. L0H2→L1H6 jumps from 0.042 → 0.608. L0H6→L1H6: 0.043 → 0.600. Standard metric is misleading — it is dominated by weight norm, not content-geometry alignment.

**Blog point**: current circuit analysis uses the wrong inner product. Inserting G (the content geometry) changes which head pairs appear to communicate — corr=0.32 means the standard metric is ~noise for identifying content-routing channels.

---

## Exp12 Results (2026-03-18) — BOS-cleaned, Sloppified Basis

Script: `notebooks/post4_qk_metric/scratch/run_exp12_sloppified_basis.py`

- BOS mass on raw G basis vectors: only directions 0 (0.051) and 1 (0.144) had meaningful BOS projection. Now zeroed.
- Sloppified spectrum (G_crude eigenvalues in induction subspace): 0.25, 0.147, 0.110, 0.092... log10 range=1.26. Not very many decades — only ~factor of 18 top to bottom. Mildly non-flat, enough to justify the ordering, but don't oversell "sloppy" framing (theoretical biology sloppy model spans 3-5 decades). Main point: spectrum is not flat, there is differential importance across directions.
- Per-head sloppiness in sloppified basis: L0 heads have FLAT G coefficients (negative or near-zero sloppiness). L1 heads are sloppy. Pattern: L0 heads are G-agnostic; L1 heads selectively engage G modes.
- Key insight: sloppiness of {a^h_i} = property of head's projection onto shared geometry, not G itself.

## Exp13 Results (2026-03-18) — JAD Basis

Script: `notebooks/post4_qk_metric/scratch/run_exp13_jad_basis.py`

Find V that maximizes sum_h ||diag(V^T G^h V)||_F^2 simultaneously (Jacobi sweeps).
- **JAD-B recovers prev-token head B basis**: top-4 principal angle cosines = 0.846, 0.774, 0.531, 0.328 (random baseline ≈0.18). Routing geometry IS genuinely shared and JAD finds it unsupervised.
- JAD-G weaker match to induction head G: top cosines = 0.701, 0.475, 0.431, 0.339. G is more head-specific.
- JAD-G coefficient sloppiness: L0 heads more sloppy (L0H4: 1.58, L0H7: 1.35) — JAD pulled out directions L0 heads specialize in. L1 heads flatter.
- **Blog: B routing geometry is shared and recoverable unsupervised via JAD.**

## Exp14 Results (2026-03-18) — Token-Token Similarity Under G and B

Script: `notebooks/post4_qk_metric/scratch/run_exp14_token_similarity.py`

G-weighted similarity W_E[i]^T G W_E[j]: **top pairs are all digit pairs** (3↔4, 5↔6, 7↔8...). G says digits are the most content-similar tokens — makes sense for induction/copying.

B-directed routing W_E[i]^T B W_E[j] (antisymmetric, no activations needed):
- '#' → 'to', 'of', 'and' (code comment token routes toward connective words)
- '\\' → 'and', '-' (escape char routes toward connectives)
- ';' → 't', '(' → 'and'/'or' (punctuation routes toward continuation tokens)
- Negative direction: 'that' → 'of' (discourse word routes toward preposition)

Pure weight geometry — no forward pass needed. B gives a directed graph on tokens.

**Open question on unsupervised G/B**: ICA/SAE-like sparsifying methods seem like the right approach. JAD (joint diagonalization) is one; sparse coding on the space of bilinear forms is another. Goal: find G/B basis where each head's coefficient profile is maximally sparse, not just sloppy.

## Exp15 Results (2026-03-18) — Effective Channel Rank

Script: `notebooks/post4_qk_metric/scratch/run_exp15_channel_rank.py`

G-weighted K-comp matrix M = W_K[1,6].T @ |G| @ W_O[0,3].T (64×64).
SVD of M gives singular values → effective rank = (sum σ)^2 / sum σ^2.

- L0H3→L1H6 standard effective rank: 38.7, stable rank: 17.4
- L0H3→L1H6 G-weighted effective rank: 34.5, stable rank: **14.1**
- Top-5 SVs (G-weighted): 1.47, 1.38, 1.37, 1.34, 1.31 — gently decaying, not a single bottleneck

**The induction circuit communicates through ~14 effective modes** — rich multi-mode channel, not a single directed pipe. Consistent with exp14 (G top pairs = semantic clusters; different categories use different modes).

L0H4 (BOS head): G-weighted rank 17-22 for all L0H4→L1Hx pairs — genuinely low-rank, but BOS-mediated. High kcomp_G for L0H4→L1H3 (0.133) is BOS noise.

---

## Post 4 Outline (revised 2026-03-18)

**Arc: motivate → toy example → what we learn → content/compute → implications for CLT/dictionary learning**
Unsupervised discovery deferred to Post 5.

**Section 1 — Motivation: why do we need G and B?**

Step 1 — Elhage's own framework undermines standard K-comp:
- Elhage: residual stream has NO privileged basis. MLP neurons do (ReLU picks out individual neurons); residual stream doesn't (any rotation absorbed into weights).
- But standard K-comp ||W_QK W_OV||_F / (||W_QK||_F ||W_OV||_F) uses the Frobenius/Euclidean inner product — a choice of basis. Inconsistent with Elhage's own argument.

Step 2 — The canonical pairing ≠ inner product (key mathematical point):
- In row-vector convention: residual stream x ∈ V* (covector). Weight matrix columns live in head space (not V*).
- x @ W_Q, x @ W_K, x @ W_V: canonical pairing V* × (head space) → R. Well-defined, no metric needed. This is how heads read — always available.
- But this pairing only tells you "how much does state x activate weight column w." It does NOT let you compare two residual stream states x and y to each other.
- Comparing two states requires a bilinear form on V* × V* → R — extra structure, not provided by the canonical pairing.
- Frobenius K-comp implicitly uses the Euclidean metric on V* — unjustified, no privileged basis.

Step 3 — G is the natural fix:
- The attention score x_q W_QK x_k^T IS a bilinear form on V* × V* → R — it compares two residual stream covectors directly.
- W_QK^{μν} = W_Q^μ_α W_K^ν_α — dot product in head space. No choice involved: attention computes Q·K by definition. The head-space contraction via δ^{αβ} IS dot-product attention, not an assumption.
- G^{μν} = sym(W_QK^{μν}) is the metric on V* induced by the standard dot-product attention (h=δ in head space). Nothing stops you from using a non-Euclidean h^{αβ} in head space — W_QK^{μν} = W_Q^μ_α h^{αβ} W_K^ν_β — which would give a different G. The 1/sqrt(d_head) scaling is already a non-trivial scalar h. So G is "the metric this model uses given h=δ," not uniquely correct in some absolute sense.
- Still decisively better than Euclidean K-comp in d_model, which uses δ in d_model — no connection to the computation whatsoever. G at least reflects the actual head-space contraction the model performs.
- **Future direction**: h^{αβ} in head space is itself inferrable from data — exactly the same problem as inferring G on V*. The model may have learned a non-Euclidean head-space metric, and the "true" G would be W_Q^μ_α h^{αβ} W_K^ν_β for that learned h. Inferring h (what head-space metric makes attention patterns most interpretable?) is a natural next-level extension of the G+B program.
- **Caution — territory vs map**: h underdetermined means G is only defined relative to a choice of h. This undermines any strong "territory" claim (G is stable across seeds because it reflects language structure). What you'd actually be measuring is stability of the h=δ projection of language geometry, not the geometry itself. For Post 4: stick to usefulness claims only, avoid territory language. Territory claim requires either (a) arguing h=δ is privileged for this architecture, or (b) jointly inferring h and G — harder, less well-defined. Table this for later.
- W_QK = G + B exactly. G = content similarity (symmetric, (2,0) tensor). B = directed routing (antisymmetric 2-form on V*). Both already there in every forward pass.
- Whether making G explicit is USEFUL is the empirical question — that's what the rest of the post shows.

**Section 2 — Toy example: build G and B by hand for the 2-layer induction circuit**
- Model: attn-only-2l (2L × 8H, d_model=512). Clean: no MLPs, K-composition is everything.
- Identify reference heads by activation (not weight heuristics): L1H6 = induction (score 0.604), L0H3 = prev-token (score 0.488). Weight heuristics give wrong answer.
- G basis = top eigenvectors of G^{L1H6} = sym(W_QK^{L1H6})
- B basis = singular vector pairs of B^{L0H3} = anti(W_QK^{L0H3})
- Housekeeping: BOS cleaning (project out v_BOS — BOS is a sink token, doesn't participate in content matching); sloppification (diagonalize G_crude in subspace → collective importance ordering)
- Result: coefficient table. In this basis, every head's type is immediately legible. L1H6: G_total=16.4 (content head). L0H3: B_total=11.2 (routing head). No activations needed.

**Section 3 — What we learn from the geometric picture**
- **Token similarity**: S_G[i,j] = W_E[i]^T G W_E[j] — which tokens does the attention mechanism treat as content-similar? Top pairs: digit clusters (3↔4, 5↔6...). S_B[i,j] = W_E[i]^T B W_E[j] — directed routing preferences, no activations. Pure weight geometry.
- **G-corrected K-composition**: Standard metric uses implicit identity. G-corrected uses the actual content geometry. L0H3→L1H6 sharpens; BOS-head confounds (L0H4, L1H7) suppressed naturally.
- **Effective channel rank**: induction circuit communicates through ~14 effective modes (stable rank). Rich multi-mode semantic channel — different token categories (digits, punctuation, etc.) use different modes.
- **Copy heads and skip trigrams**: S_G directly answers "which token does this head treat as matching which other?" Copy head = diagonal S_G. Skip trigram = off-diagonal cluster. Clean geometric prediction.

**Section 4 — Content vs compute distinction**
- G+B framing is just algebra until this: where do G and B actually live in d_model space?
- Empirical: G top eigenvectors have W_E projection mass ~0.69 (mostly token-identity space). B top modes: ~0.003 (outside W_E — compute space). NOT implied by the algebra — falsifiable prediction confirmed.
- BUT: G is not purely content. ~0.31 is compute-space. This is necessary: for K-composition to work, head B must be able to attend to head A's OUTPUT (a processed representation, not a raw token). G must span compute directions to enable cross-layer communication.
- W_V: W_E mass ~0.856 — reads almost entirely from content space. So the value pathway is mostly about token identity.
- W_O: mixed — writes both content and processed representations.
- Picture: G spans the whole residual stream geometry (content + compute). W_V reads from content. B routes via compute. Each head's expansion in G+B basis tells you exactly how it participates in this structure.

**Section 5 — Why this matters: CLT gaps and scaled dictionary learning**
- CLT attribution graphs build edges by ablating activations. They show what each head contributes — but they freeze attention patterns as given. The QK side (why a head attends where it does) is invisible.
- G fills part of the gap: G tells you which tokens are content-similar in this head's metric. High S_G[q,k] predicts high attention weight. This is the mechanistic explanation CLT graphs currently lack.
- B fills the other part: B tells you the structural routing preference — "this head prefers to attend leftward / to adjacent positions" independent of content. CLT has no representation of this at all.
- For scaled dictionary learning (SAEs, CLTs on large models): SAE features live in W_E space (content). They are good at explaining OV circuits and the G part of W_QK. But B's top modes are OUTSIDE W_E space — they are systematically invisible to current feature dictionaries. Knowing B identifies exactly the blind spot.
- Implication: annotating CLT graph edges with G-contribution vs B-contribution would qualitatively upgrade the graph — separating "this edge fires because of content matching" from "this edge fires because of positional routing."

---

## Post 5 Outline (future) — Unsupervised G+B

**The question**: Can we recover the same G and B without knowing which heads are induction/prev-token?

**What we tried (JAD)**:
- JAD-B: recovered prev-token routing geometry (top cosines 0.85, 0.77) — routing IS genuinely shared, JAD finds it
- JAD-G: weaker (top cosine 0.70) — content geometry more head-specific. JAD achieves diagonality, not sparsity.
- Key distinction: JAD ≠ sparsification. Good basis for interpretability requires SPARSE coefficient profiles, not just diagonal ones.

**The right approach: logit-space bilinear factorization**
1. For each input, compute pre-softmax logits a^h(q,k) per head
2. Symmetrize: a^sym(i,j) = [a(i,j) + a(j,i)] / 2 → this is the G-contribution
3. Antisymmetrize: a^anti(i,j) = [a(i,j) - a(j,i)] / 2 → B-contribution
4. For each head h: a^sym^h(i,j) = x_i^T G^h x_j — bilinear form in embeddings
5. Find shared G: minimize sum_h ||A^sym^h - X G^h X^T||_F^2 where G^h = sum_k a^h_k v_k v_k^T shares basis {v_k}
6. This is multi-view bilinear factorization — same concept as multi-view CCA / Shared Response Model (Chen et al. 2015)

**Softmax nonlinearity helps**: softmax makes attention patterns highly non-Gaussian (sharp, peaked). If working in pre-softmax logit space, nonlinearity doesn't appear — but it sharpens ICA signal if working in pattern space.

**Predicted result**: shared {v_k} from logit factorization should match hand-picked basis (L1H6/L0H3) up to rotation in eigenspace. This is the test. If it matches → G and B are genuinely shared objects recoverable without supervision.

### First-pass: use diagnostic token sequences as probes

Designed sequences give you G and B directly from logits, without full bilinear factorization.

**Finding G with repeated sequences:**
- Run repeated random sequence `[A B C ... A B C ...]` (as in exp10/exp11)
- For induction head L1H6 at position `dest` in the second half: logit to `dest - seq_len + 1` (same token, first half) is maximally elevated
- This is the G signal: query_t and key_t are the same token → G-contribution maximized
- Collect `a(dest, induction_src)` across many token types → samples from `x_t^T G x_t` (diagonal of G in token space)
- Varying token pairs: `a(dest_t, src_s)` for t ≠ s gives off-diagonal G entries → can reconstruct full G by fitting symmetric bilinear form to these logit samples

**Finding B with prev-token sequences:**
- On any sequence, L0H3 attends to the previous position regardless of content
- Antisymmetrize adjacent-pair logits: `[a(i, i-1) - a(i-1, i)] / 2`
- G cancels (symmetric) → residual is B-contribution: `x_i^T B x_{i-1}`
- Vary token identities at adjacent positions across many sequences → samples from B in token space
- Fit antisymmetric bilinear form to recover B

**The algorithm (supervised first-pass):**
1. Repeated sequences → L1H6 logits → fit G (symmetric bilinear form)
2. Random sequences → L0H3 logits → antisymmetrize → fit B (antisymmetric bilinear form)
3. Validate: compare to weight-based G = (W_QK + W_QK^T)/2 computed directly. Should match up to noise.

**Why this is a good first pass:**
- Uses designed probes that maximally activate the known circuit (like exp10)
- No need to assume which directions are content vs compute — the sequences force the signal
- Cheap: only need a few hundred sequences, no GPU training
- The logit samples are linear in the bilinear form → fitting is convex (least squares)

**The unsupervised extension (Post 5 proper):**
- Same idea but without knowing which heads to probe
- Run many diverse sequences; for each head compute symmetric and antisymmetric logit matrices
- Cluster heads by their logit statistics → heads with matching symmetric-logit structure share a G; heads with matching antisymmetric-logit structure share a B
- This recovers G and B without knowing head types in advance

## Future Directions

1. **Post 5: Unsupervised G/B** via logit-space bilinear factorization (see outline above)
2. **SAE comparison** (exp7, TABLED): use B_crude top-2 SVs + random baseline. Hypothesis: B top modes invisible to SAE/CLT features.
3. **G-corrected Q-comp and V-comp**: exp9 only does K-comp.
4. **Cross-seed stability** (Post 5 or later): train multiple attn-only-2l seeds, compare G eigenspaces via Procrustes. Stable → territory; unstable → map.

## Status (2026-03-19): TABLED pending more convincing preliminary evidence

Peter decided to table the shared G+B research direction after the 2026-03-19 session. The per-head G^h and B^h decomposition is solid (exact algebra, real empirical results). The shared structure claim needs validation first. **Before resuming**: run the 3 quick checks in the IOI validation section below. If they're weak or noisy, shared G+B is probably not the right story. Also: read SAEs and CLTs more carefully to understand the gap this work is trying to fill.

---

## Session 2026-03-19: Unsupervised G+B — Loss Function, IOI Plan, Softmax Gap

### Why B (positional routing) is the crux of the whole framework

G is partially recoverable from activation-based methods. SAE features live in d_model, span content directions, overlap with G's eigenbasis. CLT can find content-matching heads by looking at what activates them. G's W_E alignment means there's redundancy between G and what activation-space methods already find.

**B is constitutionally invisible to activation-based methods.** B encodes routing preferences baked into weights: which positions this head prefers to attend to, independent of content. The prev-token head attends left not because the previous token has the right content, but because anti(W_QK) has a rotation structure that scores position j-1 higher than j. You can see the result of B in attention patterns, but you cannot separate G-driven from B-driven attention without the weight decomposition. Softmax mixes them nonlinearly and they're gone.

So the framework's real contribution isn't G (which adds precision to what activation methods already gesture at) — it's B (which is genuinely new information that activation methods cannot recover). Language has both content and positional structure. Syntax is largely positional. Induction is positional. Skip-trigrams are positional. B aligns with W_pos would mean the model learned to bake syntactic/positional structure into the antisymmetric part of W_QK, separately from semantic content in G.

---

### Finding shared G and B: the correct loss function (2026-03-19)

**Setup**: Per-head G^h = sym(W_QK^h) and B^h = anti(W_QK^h) are fixed by the weights. We find shared bases {G_i} and sparse per-head coefficients a^h_i such that G^h ≈ sum_i a^h_i G_i. Error is weighted by W_OV — approximation error in directions W_OV doesn't write to doesn't matter for composition.

**L_G = sum_{h,h'} || G^h @ W_OV^{h'} - (sum_i a^h_i G_i) @ W_OV^{h'} ||_F^2 + lambda * sum_h ||a^h||_1**

**L_B = sum_{h,h'} || B^h @ W_OV^{h'} - (sum_i b^h_i B_i) @ W_OV^{h'} ||_F^2 + mu * sum_h ||b^h||_1**

**L = L_G + L_B**

This is sparse dictionary learning / alternating least squares. Outputs: shared modes {G_i}, {B_i}, sparse per-head coefficients a^h_i, b^h_i. Each shared mode is owned by few heads; heads with similar function use similar modes.

**Key design decisions:**
- W_OV weighting: only pay for errors that affect the composition channel. This is K-composition structure already baked into the loss.
- Sparsity (L1): forces each mode to be "owned" by few heads → interpretable head-type taxonomy
- No SAE features in the loss (see below)
- The sequential cascade (prev-token → induction) shows up naturally: B^{L1H6} @ W_OV^{L0H3} is one term in the sum over adjacent layer pairs

**Why NOT to use SAE features in the objective:**
SAE features are downstream of G and B — they're activations, shaped by what G+B cause the model to do. Using them to define G conflates cause and effect. SAE features live in W_E-aligned content space; using them as the target would give G ≈ W_E W_E^T — already known. SAEs are the right POST-HOC VALIDATION tool: if G modes discovered from weights align with SAE features independently, that convergence is the striking result. Don't impose it as a constraint.

**Why "forward composition of W_OV" is not an additional term:**
The sequential cascade (layer l W_OV → layer l+1 W_QK) is K-composition — already in L_G and L_B via the W_OV weighting. I initially proposed a separate L_cross term, but this was just K-comp with G and B relabeled. V-composition (W_OV^{l+1} @ W_OV^l) is a separate thing that doesn't involve G or B at all. No additional term needed.

**Sparsifying B — different algebra than G:**
B^h is antisymmetric, so v^T B^h v = 0 for any v — the diagonal trick for G (a^h_i = v_i^T G^h v_i) gives zero. B's natural structure is rotation planes: B^h = sum_k b^h_k (u_k v_k^T - v_k u_k^T) where (u_k, v_k) pairs span 2D rotation planes. Sparsifying B means finding a basis of rotation planes where per-head B^h is concentrated in few planes. JAD for antisymmetric matrices is a different algorithm. The loss L_B above handles this correctly because || B^h @ W_OV^{h'} - ... ||_F^2 doesn't require diagonal structure.

---

### The softmax gap: what weight-only loss cannot capture

The prev-token head example exposes the fundamental limitation:

B^{L0H3} routes attention to position k-1 (via rotation structure of anti(W_QK)). Softmax selects position k-1 with high weight. W_OV^{L0H3} then writes the content at k-1 to the residual. G^{L1H6} reads that content via K-composition.

**The path B^{L0H3} → softmax → W_OV^{L0H3} → G^{L1H6} is invisible to the weight-only loss.**

After softmax, B's information is consumed: it determined which values got high weight, but what's written to the residual is the content at k-1, not B itself. The residual stream can't tell you it was B that caused position k-1 to win. The weight-only loss (L_G + L_B) captures K-composition channel capacity (can G^{L1H6} read from W_OV^{L0H3} in principle). It does NOT capture how B^{L0H3} determines what flows through W_OV^{L0H3}.

To capture the softmax-mediated path you need activations — run the model, compute actual attention patterns, measure how well B^{L0H3}'s rotation plane predicts which position gets attended to. That's the positional regression approach. The weight-only loss and the activation-based analysis are measuring different things and both are needed.

---

### Rate-distortion / noisy channel interpretation (2026-03-19)

The setup is naturally information-theoretic:

- **Channel**: B^{L0H3} selects which position to route (prefer k-1). Softmax makes it stochastic across inputs — different tokens at k-1 mean different content flows through W_OV even with same routing weights. G^{L1H6} is the decoder.
- **Rate**: how sharply B concentrates attention — entropy of the softmax distribution B induces. High B magnitude → low entropy → high rate channel, reliably selects k-1.
- **Distortion**: how much G^{L1H6} fails to recover the relevant content from what W_OV wrote.
- **The noisy channel IS the softmax**: B encodes the desired routing, softmax + variable input content is the noise.

**Making it tractable with Gaussian approximation:**
If token embeddings are modeled as Gaussian, softmax attention becomes approximately a Gaussian noisy channel. Mutual information I(token at k-1; G^{L1H6} reads) has a closed form in terms of weight matrices alone — no forward passes needed. The SNR of the channel is approximately:

(top singular value of B^{L0H3} routing direction) / (interference from other positions) × (how well G^{L1H6} @ W_OV^{L0H3} recovers that direction)

This gives a weight-only approximation to the activation-based term, grounded in information theory rather than being ad hoc. The Gaussian approximation is rough — real token distributions aren't Gaussian — but it's a principled starting point.

**Connection to the loss**: weight-only L_G + L_B captures channel capacity. The rate-distortion term approximates actual information flow. Adding the Gaussian-approximation rate-distortion term to L would close part of the softmax gap without requiring forward passes.

---

### IOI on GPT-2: empirical validation plan (2026-03-19)

IOI is ideal because head types are labeled by Wang et al. 2022. G+B makes strong predictions:

**Predictions:**
- Name mover heads (9.6, 9.9, 10.0): G-dominated. Attend by content (find the name token). ||G^h||_F / ||B^h||_F high. G^h top modes have high W_E mass. Token similarity S_G = W_E G^h W_E^T clusters name tokens (Mary, John, Tom) together.
- Prev-token heads: B-dominated, low W_E mass in G modes.
- S-inhibition heads (7.3, 8.6, 8.10): mixed — content + routing. Non-trivial B capturing positional preference toward S positions vs IO positions.
- Duplicate token heads: G-dominated. G^h scores same-token pairs highly in S_G.

**Easiest experiments (no forward passes):**
1. ||G^h||_F / ||B^h||_F ratio for all 144 heads. Do known functional types separate?
2. W_E mass per head: name movers should have high G W_E mass; S-inhibition should have more compute structure.
3. S_G name token clustering: compute S_G = W_E G^h W_E^T for name mover heads vs random heads. Name tokens should cluster under name movers specifically.

**Activation-based:**
4. Positional regression on S-inhibition (7.3, 8.6, 8.10): run IOI sentences, extract antisymmetric logits, regress on (pos_S - pos_IO). Tests whether B captures the routing mechanism directly.

**Finding shared G and B in GPT-2 (harder):**
Sparsification is harder in a 12-layer model — sharing G across layers is a strong claim (layer 0 sees embeddings, layer 9 sees processed representations). Better approaches:
- Per-layer G_crude: average within each layer. 12 separate shared G's. Trivial.
- IOI-circuit G_crude: average over the ~12 known IOI circuit heads only. One matrix.
- Type-supervised: average G^h across name movers → G_content; average B^h across S-inhibition → B_routing. Functionally-meaningful shared structure within head type groups.
- SVD of stacked G^h: flatten each G^h (d_model^2 vector), stack IOI circuit heads, SVD. Top singular vectors = principal G modes shared across the circuit. 5 lines of numpy.

Cross-layer shared G (universal basis across all 12 layers) remains the hard open question. Within-layer and within-type are tractable first steps.

---

### Validation criteria for meaningful shared G and B

Three independent checks, none forced by the loss:
1. **Weight alignment**: do G_i top modes align with W_E? Do B_i rotation planes align with W_pos? Pure weight geometry check.
2. **Head type clustering**: do functionally-similar heads cluster? Name movers in IOI all have high a^h_i for the same i. S-inhibition heads for the same {B_i}.
3. **Rate-distortion SNR**: for B_i routing modes, does the Gaussian-approximation channel capacity match which circuits actually work?

If all three converge without being forced by the loss → evidence the shared G+B is real structure, not a fitting artifact. Check stability across random initializations (dictionary learning has local optima).

---

## Open Questions
1. **Cross-seed stability**: is G_crude stable across differently-initialized GPT-2 small models? If yes → G captures language geometry (territory). If no → it's a training artifact (map). This is the key test of the "territory" claim.
2. **SAE alignment** (exp7 run, TABLED): Null result — G mean max cos=0.358, B mean max cos=0.375, nearly identical. Two problems to fix before re-running:
   - Wrong B directions: stacked per-head B singular vectors have W_E content; need B_crude top modes (W_E mass ~0.003) which are the truly shared routing directions
   - Missing random baseline: 24576 SAE features in d_model=768 is 32x overcomplete; any direction finds a nearby SAE feature. Need random direction baseline to calibrate.
   - Script: `run_exp7_sae_gpt2.py` (py311 env). Fix: use B_crude top-2 SVs + add random baseline + compare W_E directions as upper bound.
3. **L0H4 anomaly**: what is L0H4 doing? Both G and B are concentrated (G_top1=23.5%). Appears as anomaly in B K-comp heatmap too.
4. **Interpretation of G negative eigenvalues**: "repulsor" directions — tokens similar in these directions avoid attending to each other (S-Inhibition type behavior?)
5. **G-corrected Q-comp and V-comp**: exp9 only does K-comp. Apply same correction to Q and V composition metrics.
6. **IOI experiments**: G/B ratio per head, W_E mass check, S_G name clustering, B positional regression on S-inhibition heads. All runnable on GPT-2 small with existing machinery.
7. **Rate-distortion term**: add Gaussian-approximation noisy channel term to L to partially close the softmax gap. SNR = top B singular value × G^{l+1} @ W_OV^l recovery quality. Tractable weight-only approximation.
8. **Type-supervised shared G+B for IOI**: average G^h within name mover heads, B^h within S-inhibition heads. Check if type-averaged G and B are more interpretable than G_crude.
