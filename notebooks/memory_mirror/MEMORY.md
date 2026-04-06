# Project Memory

## Instructions for Claude
- **Session start**: check Current Work date against today's. If stale (>1-2 days), ask Peter what he's working on and rewrite it. Always read and surface both **Current Work checklist** and **Persistent TODOs** at the start of every session.
- **Proactively maintain memory files** — minor updates (dates, bullets) just do; propose structural changes before making them. Keep MEMORY.md under 150 lines.
- **Sync memory_mirror** — a PostToolUse hook auto-syncs on every Edit/Write to the memory dir. Do NOT run `cp` manually — it's redundant and triggers a permission prompt.
- **Never delete .md files** (memory or repo) without asking.
- **Confirm before risky actions** — destructive git ops, pushing, deleting files.
- **No LaTeX in chat** — write math in plain text (x^i, W_QK, sum_s, etc.). LaTeX only in .md files that will be rendered. See [feedback_communication.md](feedback_communication.md).

---

## Recently Completed Work *(for Claude: use to detect stale info in detail files)*

**As of 2026-03-31, Peter has recently completed:**
- **2026-03-31**: OpenAI Early Career Cohort applied. Blurb: physics PhD → EBMs → mechanistic interp, neuroscience analogy, forward-pass diagnostics p<0.001.
- **2026-03-31**: NVIDIA Fundamental Generative AI applied (both JR2012698 biomolecular and JR2013293 image/video/science). Resume tailored: biomolecular framing, CUDA.jl added.
- **2026-04-02**: LASR Labs Summer 2026 applied (late). OpenAI Early Career Cohort applied. Research project arc sketched: induction head + reverse KL + C-reg.

- **2026-03-30**: Anthropic Fellows Program (July 2026) submitted. Strong application — retinal ganglion cell transfer story as centerpiece, safety framing around circuit formation prediction, closed with RS rejection quote.
- **2026-03-30**: BCG X AISI confirmed closed March 22 — missed window.
- **2026-03-25**: OpenAI Researcher, Interpretability application submitted. Final blurb saved in canonical tracker. Key framing: observational statistics over prompt classes — no training (unlike CLTs/SAEs), no per-head intervention (unlike patching). Unique novelty.
- **2026-03-25**: Job tracker consolidated into single canonical file: `notebooks/other_jobs/job_search_summary_march2026_new_new.md`. New roles added: OpenAI Alignment, OpenAI PhD General Track, OpenAI Early Career Cohort (draft blurb saved), Anthropic Alignment Science SF/London.
- **2026-03-19**: Anthropic RS Interpretability REJECTED. "Promising," encouraged to reapply ~1 year.
- **2026-03-18**: Post 4 direction decided — W_QK = G + B decomposition. Experiments 1–4 run in `notebooks/post4_qk_metric/scratch/`.

*(Update this section each session with newly completed work.)*

---

## Current Work — 2026-04-03
**Short-term priorities (this week):**

**Quick apply / EOI (low effort, do first):**
- Constellation — EOI form
- Sophia Sanborn (Stanford) — cold email, reference ENIGMA + shared research interests
- GI Lab UCSB (gi.ece.ucsb.edu/join-lab) — short research idea + CV. Pitch G+B or log-spacing of attention eigenvalues
- CAIS, MSR, IBM, Argonne — check if anything relevant open

**Bigger apps (do after PyTorch project):**
- NYU/Polymathic AI — research statement needed, HIGH PRIORITY
- FAR.AI — rolling, mech interp explicit
- Perplexity — $220K, welcomes physicists, resume + cover + research statement
- OpenAI Researcher, Alignment — high material reuse
- Apollo Research — EOI, revisit after PyTorch project (Python concern)

**Research / CV building (highest leverage):**
- PRR paper edits — get paper accepted
- PyTorch induction head project — public repo, directly addresses Python/coding gap. **NeurIPS main track deadline ~1 month out.** Target: 2-result paper (heat cap tracks induction score + C-reg improves circuit formation at marginal B). Worst case strong repo.
- Reading: Olsson ✓ (summary in notes), Reddy ✓ (read 2026-04-03, key findings: 2-layer attention-only, Gaussian mixture, B=burstiness, K=classes, L=32 labels, N=8 items, default B=2 K=256; metrics: ILA1, TILA2, IC accuracy), Singh, CLT methods paper

## Current Work — 2026-03-30
**Status**: Anthropic Fellows Program submitted 2026-03-30. Next: NVIDIA Fundamental Generative AI application.

**2026-03-30 session**: Completed and submitted Anthropic Fellows application. Key framing: retinal ganglion cell → prompt contrast transfer story as centerpiece. Safety angle: predicting circuit formation from training data statistics. Closed with RS rejection quote. BCG X AISI confirmed closed March 22 — missed it. NVIDIA New College Grad 2026 posting deadline passed Feb 7 but still worth applying.

**IMPORTANT framing (carry forward)**: Posts 1-2 diagnostics = **scalable circuit head identification** via observational prompt-class statistics. NOT the QK circuit gap (that would require G+B). CLTs/SAEs require training; patching requires intervention; Peter's approach requires neither. This is the core novelty.

**Session start checklist**:
1. **NVIDIA Fundamental Generative AI** — apply next, deadline passed but try anyway. Lead with temp tuning / generative model science, not interp.
2. **PRR paper edits** — protein sector paper peer review revisions
3. **BCG X AISI Postdoc** — CLOSED March 22. Monitor for future cohorts.
4. **Paper reading plan** — CLT methods paper, Biology of LLM, Wang 2022, Cunningham et al. SAEs (2309.08600), Olsson et al. 2022 (induction heads), Dario's "Machines of Loving Grace" + "The Urgency of Interpretability"
5. **Python/PyTorch/TransformerLens brush-up** — make a concrete practice plan
6. **Blog** — Post 3 still unwritten; G+B tabled

---

## Persistent TODOs

### Applications
- **✅ OpenAI Researcher, Interpretability** — submitted 2026-03-25
- **✅ Anthropic Fellows Program (July 2026)** — submitted 2026-03-30. Confirmation received. Expect response in 4-6 weeks.
- **BCG X AISI Postdoc** — HIGH PRIORITY, not yet applied
- **✅ OpenAI Early Career Cohort** — applied 2026-03-31 (starts June 3)
- **✅ LASR Labs Summer 2026** — applied 2026-04-02, late submission with note
- **NYU/Polymathic AI Postdoc** — HIGH PRIORITY, rolling, mech interp of scientific foundation models
- **OpenAI Researcher, Alignment** — strong secondary, high material reuse
- **Stanford ENIGMA** — formal role requires 2+ years post-PhD, skip for now. Cold email Sophia Sanborn (now at Stanford) directly — geometric/topological interp overlaps well.
- **Geometric Intelligence Lab (UCSB)** — open positions, wants short research idea. Low effort. gi.ece.ucsb.edu/join-lab. Pitch G+B decomposition or log-spacing of attention eigenvalues (sloppy models / storage capacity angle).
- **UK AISI, FAR.AI, Perplexity, Salesforce**: all pending
- **Anthropic RS**: REJECTED 2026-03-19. Reapply ~early 2027.
- **Canonical tracker**: `notebooks/other_jobs/job_search_summary_march2026_new_new.md`

### Blog Posts
- **Post 1**: `_posts/2026-02-17-why-softmax.md` — LIVE
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
- [anthropic-application.md](anthropic-application.md) — RS Interpretability application (rejected March 2026); final pass checklist
- [anthropic-fellows-app.md](anthropic-fellows-app.md) — Fellows Program (July 2026) essay bullets, framing decisions, status
- [research_ideas.md](research_ideas.md) — all ideas, backburner, pointers to detail files
- [idea_qk_metric.md](idea_qk_metric.md) — W_QK = G + B, experiments 1–4 results
- [idea_alternating_attention.md](idea_alternating_attention.md)
- **Canonical notation**: `notebooks/tensor_notation/tensor_notation_settled.md`

## Quick Reference
- **Python env**: TransformerLens + numpy/matplotlib in **conda base** `/opt/miniconda3/bin/python`
- **Elhage bug**: A=softmax(x^T W_QK x) uses column convention; correct row form: A=softmax(x W_QK x^T)
- **Post front matter**: `layout: single`, `toc: true`, `toc_sticky: true`, `mathjax: true`; `$$...$$` display, `\\(...\\)` inline. Local preview: `jserve`
- See [writing-workflow.md](writing-workflow.md), [dev-setup.md](dev-setup.md) for full details
