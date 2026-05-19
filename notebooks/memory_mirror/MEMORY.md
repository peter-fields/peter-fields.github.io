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

**As of 2026-05-18, Peter has recently completed:**
- **2026-05-18 (~5:44 AM CDT)**: **Stefan/Pivotal 2h work test submitted.** Submitted PDF writeup + Colab notebook. Task 1 (toy memorization) covered fully: Q1 accuracy across n_pairs (1.0 throughout), Q2 conceptual answer + empirical bigram-baseline finding (48% memorization via embed@unembed alone), Q3 partial. Task 2 (GPT-2 facts memorization) covered with: bar chart accuracy by category (capital best at 45%), LogitLens trajectory, *empirical discovery of " the"-prefix confound* (9 of 57 "correct" emerge at layer 0 trivially), delta plots showing rank narrows in early layers + logprob jumps at L7→L8. Q3 confounds + Q4 better-methods underdeveloped due to time. Strongest single insight: "correlated variables decrease effective number of variables" (effective-rank framing). Waiting for Stefan's response.
- **2026-05-17**: **Anthropic Fellows — advanced past initial review.** Stage 2 (60-min CodeSignal Python debugging, no AI assistance, 5 days to complete) is the active next step. References (Schwab, Ngampruetikorn, Palmer) being contacted by Constellation. See [anthropic-fellows-app.md](anthropic-fellows-app.md).
- **2026-05-04**: Constellation Astra Fellowship submitted (within AoE window). Empirical stream. Three research directions: prompt-contrast, Bauer-Bialek superposition theory, W_QK implicit metric. Mentor pitch: Owain Evans. Also eligible for Visiting Fellowship consideration.
- **2026-05-03**: Pivotal Research Fellowship submitted.
- **2026-04-22**: LASR Labs CodeSignal ML Engineering Core Assessment taken. Covered all 7 algorithms (kNN, k-Means, Decision Tree, GMM, Matrix Normalization, Bagging, Forward Prop) + Module 2 string/array problems. Got stuck on gradient descent numpy shape bug, didn't finish last problem.

**Previously:**
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

## Current Work — 2026-05-18

**Active deadline: Anthropic Fellows Stage 2 CodeSignal — 5 days from invite (~2026-05-22 EOD local).** Confirm exact deadline against CodeSignal invite email. See [anthropic-fellows-app.md](anthropic-fellows-app.md) for full prep plan, rules, and test surface area. **No AI assistance allowed during the test (no Claude Code, no Cursor, no Copilot).**

**Other immediate priorities:**
- Stefan/Pivotal 2h work test — **SUBMITTED 2026-05-18 ~5:44 AM CDT.** Awaiting Stefan's response. See [stefan-work-test-brief.md](stefan-work-test-brief.md) for what was learned.
- UK AISI Model Transparency — deadline 2026-05-24
- Python practice — central gap; Anthropic Fellows debugging assessment is the immediate proving ground

**Recently expired / unresolved:**
- BlueDot Technical AI Safety Course intensive — 4–9 May 2026 (status?)
- ARBOx4 (Oxford) — deadline May 8 (applied?)
- Blue Dot Technical AI Safety Project Sprint — deadline May 10 (applied?)

**Near term priority — make a plan for:**
- Postdoc/academic route: NYU/Polymathic AI, BCG X AISI, Geometric Intelligence Lab (UCSB), Stanford ENIGMA, cold emailing
- Funding fellowships: Coefficient Giving RFP, LTFF (rolling), Timaeus Research Fellows, BlueDot Career Transition Grant
- High school science teaching job — applying

**After BlueDot course:**
- PyTorch induction head project — highest leverage CV item
- NYU/Polymathic AI — research statement needed, HIGH PRIORITY, rolling
- CAIS, FAR.AI, Apollo — apply after PyTorch project
- Perplexity — $220K, welcomes physicists

**Research / CV building (highest leverage):**
- PRR paper edits — get paper accepted
- PyTorch induction head project — public repo, addresses Python gap
- Reading: Singh, CLT methods paper (Olsson ✓, Reddy ✓)

---

## Persistent TODOs

### Applications
- **✅ OpenAI Researcher, Interpretability** — submitted 2026-03-25
- **Anthropic Fellows Program (July 2026)** — **ADVANCED 2026-05-17.** Stage 2 = 60-min CodeSignal Python debugging, 5 days from invite, no AI assistance allowed. References being contacted now. Next stages: 5h take-home (late May) → 15-min research brainstorm interview (early June) → decisions (late June). Full notes: [anthropic-fellows-app.md](anthropic-fellows-app.md).
- **BCG X AISI Postdoc** — CLOSED. Missed window twice.
- **✅ OpenAI Early Career Cohort** — applied 2026-03-31 (starts June 3)
- **✅ LASR Labs Summer 2026** — **CodeSignal taken 2026-04-22** (went so-so). Next: Airtable paper critique from abstract (AI safety reasoning, arrives soon). Interview invitations expected early May.
- **NYU/Polymathic AI Postdoc** — HIGH PRIORITY, rolling, mech interp of scientific foundation models
- **Argonne National Laboratory** — postdoc, check for AI/ML/theory openings
- **Fermilab** — postdoc, check for AI/ML/theory openings
- **OpenAI Researcher, Alignment** — strong secondary, high material reuse
- **Stanford ENIGMA / Sophia Sanborn** — postponed for now. Cold email when stronger Python/blog presence. Geometric/topological interp overlaps well.
- **Geometric Intelligence Lab (UCSB)** — postponed. Strategy: write toy model + blog on sloppy models / log-spaced attention eigenvalues (possibly leveraging SAEs or CLTs), then reach out. Separate from PyTorch induction head project. gi.ece.ucsb.edu/join-lab.
- **CAIS Research Scientist** — apply after PyTorch project (Python gap is main concern). safe.ai/careers, rolling.
- **IBM Goldstine Fellowship** — flag in fall 2026. Deadline Dec 31, 2026. Math of AI focus, partial fit. gpfellow@ibm.com, academicjobsonline.org.
- **✅ Constellation Astra Fellowship** — submitted 2026-05-04 (AoE). Empirical stream, Owain Evans mentor pitch. Also eligible for Visiting Fellowship.
- **FAR.AI Research Scientist** — strong fit (mech interp explicit, SAE critique angle). Apply after PyTorch project — 72-min coding assessment in process.
- **UK AISI Research Engineer/Scientist – Model Transparency** — London, due **2026-05-24**. Salary £65–145K + 28.97% pension. Strong fit: mech interp, model auditing. Python/PyTorch gap is a concern for RE track; RS track more viable. URL: https://job-boards.eu.greenhouse.io/aisi/jobs/4848454101
- **Blue Dot Technical AI Safety Project Sprint** — due **2026-05-10**. 30hrs, work with AI safety expert, publish blog post. Prioritizes Blue Dot course grads (you'll be one). Rapid Grants available for compute. https://bluedot.org/courses/technical-ai-safety-project
- **MATS Scholar** — apply at matsprogram.org/apply. Rolling.
- **ARC (Alignment Research Center)** — visiting researcher positions, 10 weeks flexible. Check alignment.org/hiring for current openings.
- **LawZero** — AI Safety Research Scientist (ML focus), Montreal, non-profit, Scientist AI agenda. PyTorch required. https://job-boards.greenhouse.io/lawzero/jobs/4008813009
- **Timaeus Research Scientist (Theorist track)** — developmental interp, remote-first. **Deadline: May 3 — MISSED.** Check if rolling. £80–200K. https://timaeus.co/blog/updates/2026-04-09-hiring
- **Timaeus Research Fellows Program** — affiliate fellowship for experienced researchers, remote-first. Rolling deadline. Flexible time commitment, 1yr min. Strong fit: mech interp, learning theory. https://timaeus.co/blog/updates/2026-04-09-fellows
- **ARENA (Alignment Research Engineer Accelerator)** — 4-5 week intensive bootcamp, London (LISA). ARENA 8.0 closed (May 25–Jun 26). Submit EOI for future rounds. Requires strong Python — note for when coding skills improve. https://www.arena.education/
- **BlueDot Career Transition Grant** — full-time AI safety funding, no public deadline/amount. Eligible as BlueDot course participant. Requires 1-2 page proposal. Apply after completing course. https://bluedot.org/programs/career-transition-grant
- **PIBBSS × Iliad Residency** — CLOSED for 2026. EOI for future cohorts. https://princint.ai/programs/residency/
- **Coefficient Giving Technical AI Safety RFP** — $100K–$1M, rolling. Strong fit for independent mech interp research proposal. https://coefficientgiving.org/funds/navigating-transformative-ai/request-for-proposals-technical-ai-safety-research/
- **LTFF (Long-Term Future Fund)** — rolling, no deadline. $20K–$80K for 3–12 month independent research. Also Upskilling Grants $5K–$50K. https://funds.effectivealtruism.org/funds/far-future
- **SFF (Survival and Flourishing Fund)** — 2026 round closed. Watch for next round. https://survivalandflourishing.fund/
- **PIBBSS Fellowship** — EOI only (2026 closed, EOI for 2027). 3-month interdisciplinary fellowship, London, $4K/month + $1K housing. Strong fit: physics/complex systems background. https://princint.ai/programs/fellowship/
- **ERA:AI Fellowship (Cambridge)** — EOI only (Summer 2026 closed). 10-week research fellowship, Cambridge UK, £34K prorated + housing + meals. Talent-first, no formal credentials required. https://erafellowship.org/fellowship
- **✅ Pivotal Research Fellowship** — submitted 2026-05-03. 9-week, London (LISA). £6–8K stipend + £2K housing + meals.
- **ARBOx4 (Oxford AI Safety Initiative)** — 2-week intensive bootcamp, Oxford, June 28–July 10. Free, housing+meals provided. Technical stream: mech interp, alignment. **Deadline: May 8.** https://oaisi.org/arbox-4
- **London Institute for Safe AI** — membership application, low priority, no deadline. https://airtable.com/appjWv2IVtAvZ0MtD/pagPHOmSov1EIZ91H/form
- **Perplexity, Salesforce**: pending
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
- [user_profile.md](user_profile.md) — Peter's background, Python gaps, working style
- [lit-review.md](lit-review.md) — novelty claims, related work, citations
- [circuit-discovery-theory.md](circuit-discovery-theory.md) — circularity problem, ICA
- [posts-arc.md](posts-arc.md) — full post arc details
- [anthropic-application.md](anthropic-application.md) — RS Interpretability application (rejected March 2026); final pass checklist
- [anthropic-fellows-app.md](anthropic-fellows-app.md) — Fellows Program (July 2026) submitted essays, framing decisions, interview prep, status
- [lasr-app.md](lasr-app.md) — LASR Labs Summer 2026 submitted answers, assessment status
- [bluedot-app.md](bluedot-app.md) — Blue Dot Technical AI Safety Course application, **submitted 2026-04-26**. Intensive: 4–9 May 2026.
- [pivotal-app.md](pivotal-app.md) — Pivotal Research Fellowship application, submitted 2026-05-03. CV bullets, risk ranking, mentor responses (Stefan Heimersheim, Logan & Thomas), privileged bases answer, ambitious goal.
- [stefan-work-test-brief.md](stefan-work-test-brief.md) — Quick-ref brief for the 2h Stefan/Pivotal work test (due Mon 2026-05-18 6:59 AM CDT). Task constraints, red-team checklist, what Claude should/shouldn't do. **Reload at start of work-test session.**
- [astra-app.md](astra-app.md) — Constellation Astra Fellowship application, submitted 2026-05-04 (within AoE window). Empirical stream. Three directions: prompt-contrast, Bauer-Bialek superposition, W_QK implicit metric. Mentor: Owain Evans.
- [python-practice-plan.md](python-practice-plan.md) — Daily morning coding practice plan: numpy/pandas + four-week interview prep in parallel, then AI from scratch, then PyTorch
- [resume-general.md](resume-general.md) — General-purpose resume: finalized bullets, skills, pending items; canonical files in `notebooks/other_jobs/general_resumes/`
- [research_ideas.md](research_ideas.md) — all ideas, backburner, pointers to detail files
- [idea_qk_metric.md](idea_qk_metric.md) — W_QK = G + B, experiments 1–4 results
- [idea_alternating_attention.md](idea_alternating_attention.md)
- **Canonical notation**: `notebooks/tensor_notation/tensor_notation_settled.md`

## Quick Reference
- **Python env**: TransformerLens + numpy/matplotlib in **conda base** `/opt/miniconda3/bin/python`
- **Elhage bug**: A=softmax(x^T W_QK x) uses column convention; correct row form: A=softmax(x W_QK x^T)
- **Post front matter**: `layout: single`, `toc: true`, `toc_sticky: true`, `mathjax: true`; `$$...$$` display, `\\(...\\)` inline. Local preview: `jserve`
- See [writing-workflow.md](writing-workflow.md), [dev-setup.md](dev-setup.md) for full details
