# Project Memory

## Instructions for Claude
- **Session start**: check Current Work date against today's. If stale (>1-2 days), ask Peter what he's working on and rewrite it. Always read and surface both **Current Work checklist** and **Persistent TODOs** at the start of every session.
- **Proactively maintain memory files** — minor updates (dates, bullets) just do; propose structural changes before making them. Keep MEMORY.md under 150 lines.
- **Sync memory_mirror** — after any memory file edit, cp changed file(s) to `notebooks/memory_mirror/`. Never delete from mirror without asking.
- **Never delete .md files** (memory or repo) without asking.
- **Confirm before risky actions** — destructive git ops, pushing, deleting files.
- **No LaTeX in chat** — write math in plain text (x^i, W_QK, sum_s, etc.). LaTeX only in .md files that will be rendered. See [feedback_communication.md](feedback_communication.md).

---

## Recently Completed Work *(for Claude: use to detect stale info in detail files)*

**As of 2026-03-19, Peter has recently completed:**
- **2026-03-17**: Anthropic RS Interpretability coding screen. Q1–Q3 + Q4 part 1 done fully. Q4 part 2 (induction circuit, two-hot encoding): prev-token head done, ran out of time on induction head. Q4 part 3 not reached. Waiting for results.
- **2026-03-18**: Post 4 direction decided — W_QK = G + B decomposition. Experiments 1–4 run in `notebooks/post4_qk_metric/scratch/`. Key result: B top modes in compute subspace (W_E mass ~0.003), G in token-identity subspace (~0.69).
- **2026-03-03**: Anthropic application submitted. All essay edits complete (see [anthropic-app-edits.md](anthropic-app-edits.md)). Site Priority 1 complete (about page, nav, trust signals).
- **~2026-03-05**: Post 3 experiments complete (16 experiments). Final pipeline: contrastive ICA (7/8 circuit heads), out_mag > Var_v (30x ratio, p=1.2e-5), C_diff graph (9x enrichment at 5σ). Post not yet written.

*(Update this section each session with newly completed work.)*

---

## Current Work — 2026-03-19
**Status**: Anthropic coding screen COMPLETED 2026-03-17. Waiting for results.

**Screen debrief**: Q1–Q3 + Q4 part 1 done. Q4 part 2: prev-token head done, ran out of time on induction head. Q4 part 3: never reached. Conceptual understanding correct; execution ran out of time.

**Session start checklist**:
1. **PRR paper edits** — protein sector paper received peer review; work through revisions (see `notebooks/anthropic_app/essays.md`)
2. **Review Anthropic essays** — verbal fluency on diagnostics, circuit structure, biology→interp transfer
3. **Paper reading plan** — CLT methods paper, Biology of LLM, Wang 2022 are top candidates
4. **Python/PyTorch/TransformerLens brush-up** — make a concrete practice plan
5. **Job applications** — BCG X is next highest priority
6. **Blog/research direction** — see Active Threads below

**Core research direction**: use controlled + active prompts to extract statistics identifying circuit heads. Key gap: CLT attribution graphs freeze attention weights (assume QK given); Peter's diagnostics (ΔKL, Δout_mag, contrastive ICA) operate in that gap. What W_QK structure is invisible to CLT but visible to contrastive prompting?

---

## Persistent TODOs

### Applications
- **Finish OpenAI interpretability app** — draft blurb in `notebooks/other_jobs/job_search_summary_new.md`; needs polish + submit
- **BCG X AISI Postdoc** — HIGH PRIORITY, not yet applied
- **Anthropic RS**: coding screen done 2026-03-17, waiting for results
- **Perplexity research internship** — added 2026-03-17, details TBD
- **Anthropic Fellows Program** — confirm deadline/rolling status before deciding when to apply
- **Stanford ENIGMA, UK AISI, DeepMind, Salesforce**: all pending
- Full tracker: `notebooks/other_jobs/job_search_summary_new.md`

### Blog Posts
- **Post 1**: `_posts/2026-02-17-why-softmax.md` — READY TO PUBLISH
- **Post 2**: `_posts/2026-02-24-attention-diagnostics.md` — LIVE
  - **TODO**: causal verification of L8H1/L8H11 via activation patching (flagged as novel unlabeled heads)
- **Post 3**: experiments DONE (out_mag > Var_v 30x ratio p=1.2e-5, contrastive ICA 7/8 heads, C_diff graph). Post not written. See [post3-plan.md](post3-plan.md).
  - **Open questions before writing**: (1) do ICA components match Wang et al. causal sub-circuits? (2) C_diff vs contrastive ICA — both in post or just ICA? (3) cite cICA paper (PNAS 2025) as related work
- **Post 4**: W_QK = G + B decomposition direction (2026-03-18). Experiments 1–4 done in `notebooks/post4_qk_metric/scratch/`. See [idea_qk_metric.md](idea_qk_metric.md).

### Active Threads
- **SAE comparison**: needs `conda create -n py311 python=3.11 && pip install transformer-lens sae-lens`. Use `jbloom/GPT2-Small-SAEs-Reformatted`. Hypothesis: B compute modes invisible to CLT/SAE.
- **Blog idea**: W_QK sym/anti ratio as head-type discriminator (no precedent found)
- **Blog idea**: tensor notation for Elhage 2021 — needs privileged basis argument airtight before writing
- **Personal**: build small LLM from scratch in PyTorch; Julia is primary language, PyTorch fluency is a gap
- **Site TODO**: Priority 2 next (MathJax stability, repo hygiene). Full list: [site-todo.md](site-todo.md).
- **Lit review TODOs**: 7 pending searches in [lit-review.md](lit-review.md) (functional connectivity + transformer, spectral clustering on head stats, Kim 2026 citation graph, entropic OT equivalence, etc.)

---

## Context
- Jekyll blog (Minimal Mistakes, sunset skin, MathJax) — peter-fields.github.io
- **Two remotes**: `origin` = public GitHub Pages; `private` = private backup
- **Two branches**: `backup` (default, has notebooks); `main` (stripped, public). Always work on `backup`. Publish with `./push-site.sh`. See [dev-setup.md](dev-setup.md).

## Reference
- [lit-review.md](lit-review.md) — novelty claims, related work, citations
- [circuit-discovery-theory.md](circuit-discovery-theory.md) — circularity problem, ICA
- [posts-arc.md](posts-arc.md) — full post arc details
- [anthropic-application.md](anthropic-application.md) — final pass checklist + papers to re-read before interviews
- [research_ideas.md](research_ideas.md) — all ideas, backburner, pointers to detail files
- [idea_qk_metric.md](idea_qk_metric.md) — W_QK = G + B, experiments 1–4 results
- [idea_alternating_attention.md](idea_alternating_attention.md)
- **Canonical notation**: `notebooks/tensor_notation/tensor_notation_settled.md`

## Quick Reference
- **Python env**: TransformerLens + numpy/matplotlib in **conda base** `/opt/miniconda3/bin/python`
- **Elhage bug**: A=softmax(x^T W_QK x) uses column convention; correct row form: A=softmax(x W_QK x^T)
- **Post front matter**: `layout: single`, `toc: true`, `toc_sticky: true`, `mathjax: true`; `$$...$$` display, `\\(...\\)` inline. Local preview: `jserve`
- See [writing-workflow.md](writing-workflow.md), [dev-setup.md](dev-setup.md) for full details
