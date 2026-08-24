# Project Memory

## Instructions for Claude
- **Session start**: check Current Work date vs today; if stale (>1-2 days), ask Peter and rewrite. Always surface Current Work + Persistent TODOs.
- **Proactively maintain memory** — minor updates just do; propose structural changes first. **Keep MEMORY.md under 17KB / ~150 lines** (one line per entry; detail lives in topic files).
- **Sync** — a PostToolUse hook auto-syncs the memory dir to memory_mirror. Do NOT `cp` manually.
- **Never delete .md files** without asking. **Confirm before risky actions** (destructive git, pushing, deleting).
- **No LaTeX in chat** — plain-text math (x^i, W_QK, sum_s). LaTeX only in rendered .md. See [feedback_communication.md](feedback_communication.md). Em-dashes: Peter uses them deliberately, don't strip ([feedback_em_dashes.md](feedback_em_dashes.md)).
- **Peter is intuition-led / anti-rationalist** — give plain, jargon-free explanations when he's learning outside his domain; trust his gut; don't push him to rationalize or claim engagement he lacks ([feedback_intuition_style.md](feedback_intuition_style.md)).

---

## ⏰ ACTIVE REMINDERS
- **MATS proctored-test disclosure — support form SUBMITTED 2026-06-27** (re: Applied AI Assessment; CodeSignal Copilot inadvertently ran hidden tests; Peter self-reported proactively/in good faith 6/24, offered a retake). Raj replied 6/26 asking him to fill out `matsprogram.org/support-a26` ("we'll take this into consideration when evaluating your application"); Peter submitted it 6/27. **Ball in MATS's court — no action unless they reply; delete once resolved.**- **CodeSignal "June 24" completion date — LIKELY A NON-ISSUE (timezone).** CodeSignal logged the MATS Research Taste Test as "June 24 10:33 am +12" (UTC+12) = June 23 22:33 UTC. Deadline June 23 11:59 PM AoE = June 24 11:59 UTC, so Peter finished ~13h early, on time; "June 24" is just a +12 display. Keep the email; only ping applications@matsprogram.org if MATS ever flags it. Delete this once confirmed moot.

---

## Recently Completed *(one line each; detail in topic files)*
- **2026-08-20**: **Job-search status update — MIXED.** ❌ REJECTED: **PrincInt RS** (after the Ari Brill technical round), **Iliad Fellowship**, **MATS Autumn**. 🟢 ADVANCING: **Mercatus — final SITE VISIT** (hottest; humanities/policy lane), **MITRE** (interview next wk), **Singapore + ARENA** (both final round, ⏳ waiting), **AIAF** (submitted 8/17); ✅ **Iliad Intensive (Sept) — IN.** MATS Winter auto-advance available. Fit signal: applied/public-engagement lane advancing, pure-lab track not. (Detail in Current Work "LIVE JOB PIPELINE".)
- **2026-06-29**: **PrincInt RS → advanced to recruiter screen** (Travis Snow, Impact Ops; held 7/07 — see Current Work). Remote-first role; pay band $100–250K; comp answer if pressed = "~$170–200K, flexible, fit-first" ([princint-app.md]).
- **2026-06-27**: **MATS AI-use redo DONE**; clean self-written responses resubmitted + rankings (Sharkey #1, Gary Abel, ARC). Transferable detector lesson: *condensing/de-hedging your own writing* trips AI detectors — cut whole units yourself, keep hedges ([mats-sharkey-proposal-notes.md]).
- **2026-06-24**: **MATS Autumn 2026 R2 fully submitted/wrapped** (Gary Abel biosecurity DNA-screening exercise done; all streams + assessments in — ARC work test 6/30 still pending). St. Ignatius HS-teaching app + Meta FAIR prescreen archived to memory ([index-apps-and-projects.md]).
- **2026-06-23**: MATS R2 Empirical **Research Taste Test DONE** (nanobot-memory pre-release scenario; led with training-data-prior benign hypothesis + baseline/discriminating tests). **Sharkey 300w proposal DRAFTED** (W_QK=G+B routing-dictionary idea; final text in [mats-sharkey-proposal-notes.md]). **Gary Abel screening DONE** (handled in a separate session). **ARC stream → invited to work test.**
- **2026-06-22**: **PrincInt Research Scientist SUBMITTED** ([princint-app.md]).
- **2026-06-16**: PRR revision emailed to coauthors (DWS); awaiting feedback → APS resubmit ([prr-paper-revision.md]).
- **2026-06-11**: Meta FAIR prescreen submitted; MATS **advanced to Round 2**; Anthropic Fellows + Constellation Astra **REJECTED**.
- **2026-06-01/04**: Meta FAIR postdoc applied (6/1); Pivotal/Stefan **REJECTED** (6/4).
- **2026-05-21**: LASR Summer 2026 **REJECTED**; HS science teaching applied; Anthropic Fellows Stage 2 debug test (2/6 bugs).
- **Earlier (Mar 2026)**: OpenAI Interp (submitted 3/25), OpenAI Early Career + NVIDIA GenAI (3/31), Anthropic Fellows submitted (3/30), Anthropic RS Interp **REJECTED** 3/19, Post 4 W_QK=G+B direction decided 3/18.

---

## Current Work — 2026-08-20

**🚨 Bridge income (TOP PRIORITY):** contract ENDED end of June 2026; postdocs start Sept–Jan. Backup gap-income ideas if bar work is slow: tutoring (Wyzant), Snorkel AI Project Rudder (RLHF labeling, ~$30-45/hr), adjunct, freelance tech writing, Julia/Python consulting. St. Ignatius HS-teaching app submitted 5/21, awaiting ([ignatius-teaching-app.md]).
**🍸 Bar work (chosen active bridge plan, Chicago):** resume FINALIZED 6/29 (canonical Word doc on Peter's Desktop `Fields_Peter_Bar_Resume_2026.docx` — **authoritative, do NOT regenerate/overwrite without asking**; Dock at Montrose Beach 2023 barback→server, PhD OFF / "recently finished grad school"). Certs DONE (BASSET valid to 9/3/2026 + Food Handler 6/29). Ref: **Danyel Duncan** confirmed (former AGM, now Zooba Group; (419) 297-3068, danyel@zoombagroup.com). Availability 3–4 shifts/wk nights/weekends (1 wk away in Aug). 🟢 **RESUME already SENT to lots of places — now WAITING TO HEAR BACK** (follow up if quiet; mid-afternoon 2–4pm best for walk-ins). Optional: if Maddie replies, the valuable ask is "who's hiring," not the reference.

**🎯 PrincInt Research Scientist — ❌ REJECTED (~8/19).** Advanced through the recruiter screen (Travis Snow 7/07) AND the **Ari Brill technical interview** (ML + AI-safety-knowledge discussion; heavy prep — Ari = physicist doing physics-informed interp / percolation scaling laws, "Neural Scaling Laws Rooted in the Data Distribution" arXiv:2412.07942; felt like a strong match) — but no offer. Round-1 prep sheet still on Desktop `PrincInt_Interview_CheatSheet.md`; transferable interview answers ("changed my mind" = regularization-independence; references strengths/weaknesses) reusable. ([princint-app.md])

**🔥 LIVE JOB PIPELINE (8/20) — where the momentum actually is:**
- **Mercatus Center — 🟢🟢 HOTTEST: flying Peter out for a final SITE VISIT.** Future of Scientific Discovery Emerging Scholar, 1-yr Arlington VA, $80–150K; the **humanities/policy lane** from the ZOË/Cluny/Luke-Burgis intro (app + research one-pager in [mercatus-app.md]).
- **MITRE — 🟢 interview WED AUG 26, 1:15–2:15pm EST (1 hr, Teams) w/ Dr. Evelyne Tzoukermann (Group Lead).** Role **R116623 Generative AI Engineer** — applied/IC-facing (L574 Multi-INT AI Engineering / N970 Global Intelligence), **McLean VA** (DC metro — same region as Mercatus/Arlington; the DC applied+policy lane is converging). They reached out; Peter advanced **straight to a 1-hr program-lead interview**. Likely conversational (role/fit + genAI knowledge), NOT live coding. Prep: applied genAI (RAG/fine-tuning/evals/deployment) at concept level + honest-but-confident on eng level (deep fundamentals, fast learner, Julia-primary/Python ramping). CC to pull Tzoukermann background + MITRE genAI division for tailored prep.
- **Singapore AISF — 🟢 final-round interview DONE (~8/19), ⏳ waiting** ([singapore-aisf-app.md]).
- **ARENA 9.0 — ❌ REJECTED (~8/21)** (had the final interview; no offer) ([arena-app.md]).
- **AIAF / AE Studio Alignment Fellowship — ✅ submitted 8/17, awaiting** (Sept 8–Oct 30) ([aiaf-fellowship-app.md]).
- **✅ Iliad INTENSIVE (Sept 7–Oct 2, London) — GOT IN** (the full **Iliad Fellowship was ❌ rejected**). 🔴 **Intake form PENDING — Peter holding it until Singapore decides (~8/28); Singapore would REPLACE Iliad (overlapping dates, would strongly consider taking it), while Mercatus/MITRE start AFTER so no conflict.** Sending Aron (admissions@iliad.ac) an honest heads-up 8/22 asking to wait ~1 wk (intake triggers travel/housing/visa processing). Draft ready.
- **MATS Autumn ❌ not accepted**, but a **standing auto-advance to MATS Winter Round 2** is available if he wants it (decide later).
- **Fit signal:** the pure technical-lab track (PrincInt, MATS, Iliad-fellowship) has been the hardest door; the **applied + public-engagement lane (Mercatus, MITRE, ARENA, Singapore)** is where he's advancing. Worth leaning in.

**MATS Autumn 2026 R2 — ❌ NOT ACCEPTED (~8/19).** Sharkey (#1, W_QK=G+B proposal) rejected 7/16; Gary Abel (#2) + overall = no; ARC (#3) withdrawn 7/04. **Standing offer: auto-advance to MATS Winter Round 2 if he wants — decide later** (see LIVE PIPELINE above). ([mats-round2-streams.md])

**PRR paper — 🟡 one last David lap.** Full revision + caption audit done (Fig 2 Δ→5/5 samples, Fig 3 M=71/T=2.05, A3(c) M=40; all verified). Wave ✅ + Stephanie ✅. **David (Schwab) gave recs on the OLD version → Peter incorporated them + regenerated `diff.tex` 7/17** (baseline now lives in `older versions/original_submission/` after a repo restructure — quote the space in the latexdiff path). 🔴 **Next: send David the NEW diff (point him to where his recs landed for a fast bless) → then compile bundle → APS PRR upload.** temp-tune reproducibility repo fully PUBLISHED 7/02 (all 4 appendix figs; public repo + mirror + backup in sync) ([prr-paper-revision.md]).

**Near-term:** **BlueDot Sprint — Unit 1 eval `github.com/peter-fields/syco-not` — 🏁 EXPERIMENTS COMPLETE 8/04; ⏸️ Peter ON VACATION ~1 wk (resumes ~mid-Aug), nothing left to RUN.** Matched-pair design measures sycophancy-as-miscalibration on persecution beliefs; **BONUS: Opus 5 nearly fixed Opus 4.5's failure (validates 81%→21% of unwarranted beliefs) = training-generation not scale**; control null holds all 5 (domain not format); harshness dropped (undefined/likely tone). 🔴 **On return (human-only): review 3 CC-authored final prompt sets + rubric, poke playground.ipynb §3, write up** (figures + report ready). Full detail [syco-not-project.md]; results `notes/REPORT_2026-07-30.md` + `NEXT_STEPS.md`. Part B TABLED. [bluedot-app.md]. **Notre Dame Faith & AI = PASSED/skipped** (revisit next cycle if ever). **ZOË Scholarship 🎉 AWARDED/ACCEPTED — Cluny conf, Napa July 26–28; 🔴 book flights ASAP + send confirmation to Luke** (Meritage + $1K honorarium + Mercatus intro; overlaps Iliad 7/27 → submit Iliad early; full detail + contacts [zoe-scholarship-app.md]). **Kairos GCP workshop = optional networking foot-in-door only; Generator = ❌ NOT a fit (ops, closed)** — full read + Peter's 2026-07-20 anti-field-building recalibration in [kairos-fieldbuilding-notes.md]. **Strategic: the humanities/policy lane (Cluny, Mercatus, faith-&-AI) is a distinct strong fit worth cultivating alongside the technical-lab track.** Fall residency fork below carries the next clocks.

**CV pivot — 🔵 ON BACKBURNER (2026-06-24, not off the ground yet; deprioritized for a while):** compressed-computation project (Hänni U-AND + Bauer-Bialek, mostly NumPy) was to replace the dropped PyTorch induction-head project; Stefan Heimersheim = audience ([compressed-computation-project.md]).

**Academic route (no coding tests):** NYU/Polymathic postdoc (HIGH PRIORITY, rolling); Geometric Intelligence Lab UCSB (needs toy-model blog first); cold-email academics.

**⛰️ FALL 2026 RESIDENCY FORK — pick ONE (all Sept–Dec, conflict; apps cheap + staggered → apply to all 3, decide on offers):** (a) **Iliad Fellowship** (PrincInt residency, `princint.ai/programs/residency/`) — LISA London Sept 7–Dec 4, applied math for alignment, $6K/mo allowance; **❌ FELLOWSHIP REJECTED ~8/19 → but ✅ GOT the 1-month Iliad INTENSIVE (September) instead**; best theory-fit; *distinct from the PrincInt RS role (also ❌ rejected).* (b) **Singapore AISF — 🟢 SHORTLISTED → work test ✅ (8/03) → FINAL-ROUND INTERVIEW DONE ~8/19, ⏳ WAITING.** Prep + post-mortem in `notebooks/practice/singapore_test_prep/` (`python_refresher.ipynb` = runnable basics refresher w/ 6 self-checking drills; `batches_pattern.py` = the expiry/sub-batch pattern he got stuck on). Sept 21–Dec 4, SGD 5K/mo + housing + travel + up to USD 30K compute; mentors Krampis / James Chua / Min-Yen Kan ([singapore-aisf-app.md]). (c) **ARENA 9.0 — ✅ SUBMITTED 7/12 → coding test ✅ (8/03) → final interview → ❌ REJECTED ~8/21** — LISA London Oct 5–Nov 6, PyTorch bootcamp, costs covered ([arena-app.md]). Post-mortem (Zeckendorf/non-consecutive-Fibonacci question + the "tabulate small cases before reaching for DP" lesson) in [arena-test-prep.md]; worked solution `notebooks/practice/singapore_test_prep/zeckendorf.py`.

**🎯 Next-app queue (Peter's order, set 2026-06-24; the fall fork above sits alongside this):** (1) **Iliad Fellowship** — see Fall residency fork above (apply 7/27). (2) **Coefficient Giving Career Dev & Transition Funding** (GCR fund, rolling) — doubles as bridge income, consider near-parallel with Iliad. (3) **Geometric Intelligence Lab UCSB** (Google Form) — Peter has a friend in a temp position there; **asking him for a vibe check first** before applying (warm-intro path). The "toy-model blog" prereq was Peter's own strategic read (job page states no requirements) = the paused compressed-computation write-up, NOT a hard GI Lab requirement. (4) **W. Jeffrey Johnston** (wj2.github.io; Columbia/Fusi → UCSF asst prof July 2026; representational geometry, recruiting all levels) — rolling, cold-email wjeffreyjohnston@gmail.com.

**Funding (after paper):** Coefficient Giving RFP, LTFF, Timaeus Fellows, BlueDot Career Transition Grant (all rolling).

---

## Persistent TODOs

### Applications *(canonical tracker: `notebooks/other_jobs/job_search_summary_march2026_new_new.md`)*
- ✅ **Submitted/applied**: OpenAI Interp (3/25), NVIDIA GenAI (3/31), Meta FAIR postdoc (6/1; prescreen 6/11), PrincInt RS (6/22; **screening call invited 6/29**), St. Ignatius HS teaching (5/21).
- 🟢 **Active**: ❌ ARC stream **WITHDRAWN 7/04** (declined the tight 7/15→7/22 turnaround; #3, burnt out — Sharkey/Gary Abel explicitly retained); **⛰️ Fall residency fork — pick ONE (see Current Work): Singapore AISF (✅ SUBMITTED 7/10), ARENA 9.0 (✅ SUBMITTED 7/12), Iliad (apply 7/27), AIAF/AE Studio Fellowship (✅ SUBMITTED 8/17 — Sept 8–Oct 30, same slot, conflicts with all three)**; BlueDot Project Sprint (✅ submitted 7/03; now in course — Unit 1 project chosen, see Current Work); NYU/Polymathic (rolling, HIGH). 🟡 **Undecided/maybe-skip**: Notre Dame Faith & AI (7/1, not started); Hostačov AI-safety co-living residency near Prague (July 1–22 or Jul 27–Aug 10, **~€30–60/day = a cost not income** → tension w/ bridge-income & bar-work plan; upside = networking + philosophy vibe; `hostacov.notion.site/first-summer`).
- ❌ **Rejected**: **PrincInt RS (~8/19, after Ari Brill technical round), Iliad Fellowship (~8/19 — but got the 1-mo Intensive), MATS Autumn (~8/19, Winter auto-advance available);** Anthropic RS Interp (3/19, reapply ~2027), OpenAI Early Career (not selected), LASR (5/21), Pivotal (6/4), Anthropic Fellows (6/11; July cohort — rejection email is generic/no personalized feedback, "what stands out" = exceptional research/eng + demonstrated AI-safety engagement; recommends Rohin Shah impactful-research talk + Carlini research-prioritization post + 80k job board → review at some point), Constellation Astra (6/11), BCG X AISI (missed).
- ⏸ **Dropped/postponed**: OpenAI Alignment, UK AISI Transparency (5/26); Stanford ENIGMA/Sanborn, Geometric Intelligence Lab UCSB, CAIS, FAR.AI (after PyTorch/blog).
- 📋 **Later/EOI**: Argonne, Fermilab, IBM Goldstine (Dec 2026), LawZero, Timaeus RS/Fellows, PIBBSS, ERA:AI, London Inst Safe AI, Perplexity, Salesforce; W. Jeffrey Johnston / UCSF (queue #4, cold-email). *(ARENA promoted to Active — see above.)*
- 💰 **Funding (rolling)**: Coefficient Giving RFP ($100K-1M); **Coefficient Giving Career Dev & Transition Funding (queue #2, = bridge-income angle)**; LTFF ($20-80K), SFF (closed—watch), BlueDot Career Transition Grant.

### Blog Posts
- Post 1 (why-softmax) + Post 2 (attention-diagnostics) **LIVE**.
- Post 2.1 (IOI candidate-heads patching) DRAFT skeleton — `notebooks/post2.1_candidate-heads-patching/`.
- Post 3: experiments DONE (out_mag>Var_v 30x p=1.2e-5, cICA 7/8, C_diff), unwritten ([post3-plan.md]).
- Post 4: W_QK=G+B, exps 1-4 in `notebooks/post4_qk_metric/scratch/` ([idea_qk_metric.md]).

### Active Threads
- SAE comparison: py311 env + `jbloom/GPT2-Small-SAEs-Reformatted`; hypothesis = B compute modes invisible to SAE/CLT.
- Blog ideas: W_QK sym/anti ratio as head-type discriminator; tensor notation for Elhage 2021.
- PyTorch still a gap (Julia primary); induction-head project dropped for compressed-computation.
- Site TODO: MathJax stability, repo hygiene ([site-todo.md]). Lit review: 7 pending searches ([lit-review.md]).

---

## Context
- Jekyll blog (Minimal Mistakes, sunset, MathJax) — peter-fields.github.io. Remotes: `origin`=public Pages, `private`=backup. Branches: `backup` (default, notebooks) vs `main` (stripped public). Work on `backup`, publish `./push-site.sh` ([dev-setup.md]).
- Python: TransformerLens + numpy/matplotlib in conda base `/opt/miniconda3/bin/python`.
- Peter calls the assistant "CC" (fine to use) ([user_calls_me_cc.md]).
- **Elhage bug**: correct row form is A=softmax(x W_QK x^T) (not column). Post front matter: `layout: single, toc: true, toc_sticky: true, mathjax: true`; `$$..$$` display, `\\(..\\)` inline; local preview `jserve`.

## Reference (topic files)
- **🗺️ WHERE EVERYTHING LIVES (apps, proposals, resumes, projects, paper, blog + gaps): [index-apps-and-projects.md]** — check first when locating any app essay or project file.
- Profile/identity: [user_profile.md], [user_profile_faith_philosophy.md] (Catholic faith, anti-rationalist stance, symbol-grounding project), [feedback_intuition_style.md] (intuition-led working style), [substack-physicist-catholicism.md].
- Research: [research_ideas.md], [idea_qk_metric.md] (W_QK=G+B), [idea_jlens_geometry.md] (geometric critique of Anthropic J-lens/J-space), [idea_alternating_attention.md], [compressed-computation-project.md], [circuit-discovery-theory.md], [posts-arc.md], [lit-review.md], [prr-paper-revision.md], [temp-tune-publish-workflow.md] (squash-merge PRs → sepalmer/temp-tune public main).
- MATS R2: [mats-round2-streams.md], [mats-sharkey-proposal-notes.md] (final Sharkey draft saved), [mats-openai-proposal-notes.md].
- Apps: [princint-app.md], [singapore-aisf-app.md], [arena-app.md], [aiaf-fellowship-app.md] (AE Studio; W_QK=G+B essay + Berg self-referential-processing critique), [syco-not-project.md] (BlueDot eval — Part A reproduced + v2 plan + open rubric decision), [anthropic-fellows-app.md] (⭐ has reusable reference blurbs for all 3 coauthors), [anthropic-application.md], [lasr-app.md], [bluedot-app.md], [zoe-scholarship-app.md], [mercatus-app.md] (humanities/policy lane; app + research one-pager), [kairos-fieldbuilding-notes.md], [pivotal-app.md], [astra-app.md], [stefan-work-test-brief.md], [resume-general.md].
- Practice/workflow: [python-practice-plan.md], [arena-test-prep.md] (ARENA coding-test prep + strategy, deadline Aug 2), [debug_future_practice.md], [writing-workflow.md], [dev-setup.md], [site-todo.md]. Notation: `notebooks/tensor_notation/tensor_notation_settled.md`.
