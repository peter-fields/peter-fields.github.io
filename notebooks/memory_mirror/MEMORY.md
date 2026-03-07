# Project Memory

## Instructions for Claude
- **Update Current Work every session** — compare the date in Current Work against today's date (available via system context). If more than a day or two old, ask Peter what he's working on and rewrite it.
- **Proactively maintain these memory files** — consolidate, trim, or restructure as needed without being asked. Keep MEMORY.md under 150 lines.
- **Sync memory_mirror** — after any memory file update, copy the changed file(s) to `notebooks/memory_mirror/` in the repo. Never delete files from the mirror without asking. Mirror is tracked on `backup` branch, excluded from public `main`.

## Current Work (update this every session)

### peter-fields.github.io — 2026-03-06
**Goal**: Pass Anthropic coding screen (120 min, 4 Qs, Python + NumPy only, no PyTorch).
**Active files**:
- `notebooks/anthropic_app/interp_prep/nb2_attention_circuits.ipynb` — currently working through (Day 2-3 of study plan)
- `notebooks/anthropic_app/A Mathematical Framework for Transformer Circuits.html` — Elhage 2021, reading §1-4
- `notebooks/anthropic_app/interp_prep/study_plan.md` — full 7-day plan

**Recent progress (2026-03-06)**:
- Worked through tensor notation for Elhage 2021 — clarified W^T origin, W_QK as (2,0) vs W_OV as (1,1), caught Elhage convention bug in attention score formula. Notes in [tensor_notation.md](tensor_notation.md).
- Blog idea: W_QK^sym / W_QK^anti ratio as interpretability discriminator for head type.

---

## Context
- Blog for Anthropic interpretability research job application
- Jekyll + Minimal Mistakes (sunset skin), MathJax enabled
- Site: peter-fields.github.io

## Anthropic App — COMPLETE (2026-03-03)
All edits done. Papers to re-read before interviews: Circuit Tracing, Biology of LLM, Elhage 2021, Wang 2022, Kim 2026.
See [anthropic-application.md](anthropic-application.md) for final pass checklist + files to review.

## Peter's TODO (personal)
- Build a small neural net / LLM from scratch in PyTorch. Primary language is Julia; Python/PyTorch fluency is a gap.

## Three-Post Arc — see [posts-arc.md](posts-arc.md) for full details
- **Post 1**: `_posts/2026-02-17-why-softmax.md` — READY TO PUBLISH
- **Post 2**: `_posts/2026-02-24-attention-diagnostics.md` — COMPLETE AND LIVE
- **Post 3**: planned — Var_v upgrade + factor analysis + J_diff circuit graph (see [post3-plan.md](post3-plan.md))
- **Post 4**: planned — Binary RBM + MI regularization (Kyle likely co-author)

## Research Notes
- [lit-review.md](lit-review.md) — novelty claims, related work, full citation list with URLs
- [circuit-discovery-theory.md](circuit-discovery-theory.md) — circularity problem, structure hypotheses, ICA
- [tensor_notation.md](tensor_notation.md) — Elhage notation fix, index conventions, blog ideas

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
