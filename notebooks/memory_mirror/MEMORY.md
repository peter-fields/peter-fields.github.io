# Project Memory

## Instructions for Claude
- **Session start**: check Current Work date against today's. If stale (>1-2 days), ask Peter what he's working on and rewrite it. Read the active files listed there if context is needed.
- **Proactively maintain memory files** — minor updates (dates, bullets) just do; propose structural changes before making them. Keep MEMORY.md under 150 lines.
- **Sync memory_mirror** — after any memory file edit, cp changed file(s) to `notebooks/memory_mirror/`. Never delete from mirror without asking.
- **Never delete .md files** (memory or repo) without asking.
- **Confirm before risky actions** — destructive git ops, pushing, deleting files.
- **No LaTeX in chat** — VSCode window doesn't render it. Write math in plain text (x^i, W_QK, sum_s, etc.). LaTeX only in .md files that will be rendered. See [feedback_communication.md](feedback_communication.md).

## Current Work (update this every session)

### peter-fields.github.io — 2026-03-07
**Goal**: Pass Anthropic coding screen (120 min, 4 Qs, Python + NumPy only, no PyTorch).
**Active files**:
- `notebooks/anthropic_app/interp_prep/nb2_attention_circuits.ipynb` — Ex 1-4 complete, Ex 5-6 remaining
- `notebooks/anthropic_app/A Mathematical Framework for Transformer Circuits.html` — Elhage 2021 full paper, readable by Claude directly
- `notebooks/anthropic_app/interp_prep/study_plan.md` — full 7-day plan
- `notebooks/tensor_notation/tensor_notation_settled.md` — canonical index notation for Elhage

**nb2 progress (2026-03-07)**:
- Ex 1 softmax ✓, Ex 2 log_softmax + cross_entropy ✓, Ex 3 softmax_jacobian ✓, Ex 4 single-head causal attention ✓
- Elhage 2021 §1-3 reading done (between Ex 4 and Ex 5)
- Next: Ex 5 multi-head attention, Ex 6 circuit matrices
- **Still to practice (2026-03-13)**: one-layer analysis (W_E W_QK W_E^T, W_E W_OV W_U, eigendecompose W_QK, identify copy heads), two-layer composition (K-composition, induction head = prev-token head + K-comp), end-to-end from raw weights → head type classification. All NumPy from scratch.

**Status (2026-03-07 end of day)**:
- Taking day off — returning 2026-03-08
- Currently reviewing tensor_notation.md for errors (Roman-left-Greek-right convention, SVD read/write primal/dual assignments, W_OV^T* argument, head composition formula)
- Do NOT treat notation as settled until review complete

**Recent (2026-03-06)**:
- Worked through tensor notation for Elhage 2021 — clarified W^T origin, W_QK as (2,0) vs W_OV as (1,1), caught Elhage convention bug. Notes in [tensor_notation.md](tensor_notation.md).
- Blog idea: W_QK^sym / W_QK^anti ratio as interpretability discriminator for head type.

**Advice from David (Stephanie's colleague, familiar with Anthropic process)**:
- Be in a comfortable, familiar place, solid internet, no distractions
- "We haven't taken tests in a while — jarring to perform under pressure"
- Practice by writing code yourself, not just reading it

---

## Context
- Jekyll blog (Minimal Mistakes, sunset skin, MathJax) hosted on GitHub Pages: peter-fields.github.io
- Purpose: Anthropic interpretability research job application — posts, notebook, essays
- **Two remotes**: `origin` = public GitHub Pages repo; `private` = private backup repo
- **Two branches**: `backup` (default, has notebooks + everything); `main` (stripped, public Pages)
- **Always work on `backup`**. Publish with `./push-site.sh` — merges backup→main, strips private notebooks, pushes to origin. See [dev-setup.md](dev-setup.md) for full git workflow.

## Anthropic App — COMPLETE (2026-03-03)
All edits done. Papers to re-read before interviews: Circuit Tracing, Biology of LLM, Elhage 2021, Wang 2022, Kim 2026.
See [anthropic-application.md](anthropic-application.md) for final pass checklist + files to review.

## Research Ideas (speculative)
- [idea_alternating_attention.md](idea_alternating_attention.md) — alternating causal/bidirectional attention; retroactive recontextualization; training curriculum based on tokens that require backward context

## Active Threads (things in flight)
- **Anthropic coding screen** — current sprint, ~7 days, NumPy/Python (see Current Work)
- **Post 3** — planned, not started; Var_v upgrade + factor analysis + J_diff circuit graph
- **Post 4** — planned, Kyle co-author; binary RBM + MI regularization
- **Site Priority 1** — about page, nav, trust signals (see [site-todo.md](site-todo.md))
- **Blog idea** — W_QK sym/anti ratio as head-type discriminator (no precedent found)
- **Blog idea** — tensor notation for Elhage 2021: x ∈ V_pos ⊗ V_feat* explains W^T, W_QK vs W_OV type distinction, Elhage convention bug. **TODO before writing**: think through privileged basis argument carefully — deepest motivation (no privileged basis in V_feat → covector declaration is substantive; V_pos having one is why Roman indices carry no primal/dual content). Needs to be airtight.
- **Personal** — build small LLM from scratch in PyTorch; Julia is primary language, PyTorch fluency is a gap

## Three-Post Arc — see [posts-arc.md](posts-arc.md) for full details
- **Post 1**: `_posts/2026-02-17-why-softmax.md` — READY TO PUBLISH
- **Post 2**: `_posts/2026-02-24-attention-diagnostics.md` — COMPLETE AND LIVE
- **Post 3**: planned — Var_v upgrade + factor analysis + J_diff circuit graph (see [post3-plan.md](post3-plan.md))
- **Post 4**: planned — Binary RBM + MI regularization (Kyle likely co-author)

## Research Notes
- [lit-review.md](lit-review.md) — novelty claims, related work, full citation list with URLs
- [circuit-discovery-theory.md](circuit-discovery-theory.md) — circularity problem, structure hypotheses, ICA
- [tensor_notation.md](tensor_notation.md) — theory, blog ideas, deeper analysis (NOT canonical notation)
- **Canonical notation**: `notebooks/tensor_notation/tensor_notation_settled.md` — index conventions, multi-layer circuits, K-composition formulas. Always check here for notation.

## Elhage Tensor Notation
- Declare x_i ∈ V_feat* (covector) → W^T follows; W_QK is (2,0), W_OV is (1,1)
- **Bug in Elhage**: A=softmax(x^T W_QK x) uses column convention; h(x)=AxW_OV^T uses row convention. Correct row form: A=softmax(x W_QK x^T).
- **Blog idea**: W_QK^sym vs W_QK^anti ratio discriminates content-matching vs directional heads (no precedent found)

## Python Environment
- TransformerLens + numpy/matplotlib: **conda base** `/opt/miniconda3/bin/python`
- Other envs (gemini-logprobs, llm-logprobs, sca) do NOT have numpy/transformer_lens

## Git Workflow — see [dev-setup.md](dev-setup.md) for Jekyll setup, server commands, gotchas
- **Always work on `backup` branch**
- `origin` → public GitHub Pages; `private` → private repo
- Daily: `git add <files> && git commit && git push` → goes to `private/backup`
- Publish: `./push-site.sh` (merges backup→main, strips notebooks, pushes origin/main)

## Post Writing — see [writing-workflow.md](writing-workflow.md) for full template + conventions
- Front matter: `layout: single`, `toc: true`, `toc_sticky: true`, `mathjax: true`, `excerpt:`
- MathJax: `$$...$$` display, `\\(...\\)` inline; `\label{}` + `\eqref{}` for cross-refs
- Local preview: `jserve` → http://127.0.0.1:4000

## Site TODO
- See [site-todo.md](site-todo.md) for full prioritized checklist
- Priority 1: About page, nav, trust signals
