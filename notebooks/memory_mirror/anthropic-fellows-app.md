---
name: anthropic-fellows-app
description: Anthropic Fellows Program (July 2026) application — settled essay bullets and key framing decisions
metadata: 
  node_type: memory
  type: project
  originSessionId: 87b87111-14b3-4b95-8b8b-2cdb5bb23448
---

# Anthropic Fellows Program — Application Notes

**Program:** July 2026 cohort, 4 months, paid ($3,850/week), Berkeley workspace
**Form:** https://airtable.com/appiuxxfhf5moRwTx/pagULi7KjbpUaOdIg/form
**Status:** **ADVANCED TO STAGE 2 — 2026-05-17.** 60-min CodeSignal Python debugging assessment incoming. 5 days from CodeSignal invite to complete (deadline ~2026-05-22 EOD local — confirm against invite email). References (Schwab, Ngampruetikorn, Palmer) being contacted by Constellation now.
**Previously:** SUBMITTED 2026-03-30; confirmation from Joe (anthropicfellows@constellation.org). 2026-04-11 automated re-invitation email (sent to all RS applicants); Peter emailed Joe 2026-04-14 to confirm application already received.

---

## Key Framing Decisions

- **Do NOT mention the RS Interpretability rejection in the essays.** Form surfaces previous application; let it. Don't lead with it.
- **Don't oversell a specific research agenda.** Pitch is: "I've done real work, here's one direction, I'm here to learn and contribute to whatever the team thinks is highest priority."
- **Code sample:** temp tuning repo only. IOI notebook was Claude-generated (Peter verified line-by-line but don't submit as code sample).
- **References:** coauthors on temp tuning paper (Schwab, Ngampruetikorn, Palmer). Pick whoever can speak most concretely to independent contributions.
- **Full-time offer:** good chance of accepting; small chance of academic pivot if AI research feels like wrong fit.

---

## Essay 1: "Why Fellows?" — SUBMITTED

My background in statistical physics and information theory makes me well-positioned for frontier AI safety research; this fellows program would actualize my potential. I have already begun to see direct evidence for this: in recent blog posts, I ported concepts from theoretical neuroscience to circuit research in mechanistic interpretability. Colleagues from my advisor's (Stephanie Palmer's) lab at UChicago recently revealed stimulus-independent structure in retinal ganglion cells' collective information processing by contrasting population response across several stimuli (doi/10.1073/pnas.2313676121). In my recent work published on my blog (peter-fields.github.io), I used a similar idea: track statistics of attention heads over different kinds of prompt classes (like tracking neural population statistics over different visual stimuli) in order to see which heads change their behavior under varying conditions and which do not (like revealing stimulus-independent information processing structure in the retina). I developed characterizations of attention-head distributions from ideas in info theory and statistical physics, and found statistically significant signatures that separated circuit attention heads from non-circuit attention heads for GPT-2 small's Indirect Object Identification circuit (p<0.001).

Research in AI safety requires both an incisive technical approach and a positive vision for AI's future—not simply a preventative and cautionary agenda. Anthropic offers all of these. Dario Amodei's essays "Machines of Loving Grace" and "The Urgency of Interpretability" testify to both with specificity. On the technical side, Anthropic has made significant developments towards understanding frontier LLMs, from induction circuit formation to Cross-Layer Transcoders and feature attribution graphs. On the safety side, Anthropic invests heavily in preventative measures (e.g. Safeguards Research Team to prevent jailbreaking, Frontier Red Team to mitigate catastrophic risk, etc.) while seeking a positive future for AI and to remain transparent—as shown by Amodei's candor in his recent interview with Ross Douthat. Learning at Anthropic through this program would develop my technical skills and intuitions through access to frontier AI technology and, more importantly, access to pioneering researchers on that frontier. My recent application to a research position on Anthropic's Interpretability team was met with this response: "We thought your application and [coding] test results were promising, and we would like to encourage you to consider applying again in a year after you've gained more experience." This fellows program would allow me to follow through on that encouragement.

---

## Essay 2: "Excited area of AI safety?" — SUBMITTED

Mechanistic Interpretability: working within this field would be a natural extension to my dissertation research. The core of my dissertation work sought to understand the relationship between generative performance, limited training data, objective function inductive bias, and implications for what this tells us about the true data-generating process. A circuit is a highly generalizable unit of computation learned from structure inherent to data. Understanding the safety implications of that circuit is akin to asking just how well that circuit generalizes to contexts unseen. Formation of induction circuits (and their ability to generalize well to in-context learning) has been shown to be robust to different choice of datasets ("In-context Learning and Induction Heads," Olsson et al.). Are there other generic properties in the statistical structure of language that may be linked to circuit formation? And how well would these circuits generalize at inference time? Another interesting direction is to extend the prompt-contrast idea from my blog posts (https://peter-fields.github.io) to larger models. I have shown that tracking shifts in per-head statistics over different prompts yields statistically significant signatures that separate circuit heads from non-circuit heads in the IOI circuit in GPT-small. (To me this is like testing what generalizable structures the model learned). This only requires a forward-pass, circumventing direct causal intervention and avoiding costly training that auxiliary methods such as CLTs require. My results are preliminary, but it would be interesting to see if different prompt classes may be used on larger models. This could potentially narrow down candidate circuit-heads that are prompt-dependent—direct causal interventions would follow.

---

## Pipeline (from 2026-05-17 advancement email)

1. ✅ Initial review (submitted 2026-03-30)
2. ✅ 90-min CodeSignal (general coding ability) — passed
3. **← YOU ARE HERE: 60-min CodeSignal (Python debugging), 5 days to complete**
4. Second review
5. 5-hour take-home research project (late May, 5-day turnaround)
6. 15-min research brainstorm interview (early June)
7. Final decisions (late June)

**Program starts:** 2026-07-20.

---

## Stage 2: 60-min CodeSignal Python Debugging Assessment

**Critical rules:**
- **No AI assistance.** No Claude Code, Cursor, Copilot. AI Overview in Google search results is OK; AI Mode is NOT. Code only inside CodeSignal (no external editor; pasting is flagged).
- **Proctored.** Webcam, microphone, government photo ID required.
- One attempt only. Score may be shared with partner programs unless Peter opts out via email to anthropicfellows@constellation.org.

**Format:**
- Existing codebase with failing unittest tests. Fix root cause so tests pass.
- 60 min, finishing early adds bonus. Don't need to pass all tests to advance.
- Read tests; use them as final word on requirements. Don't worry about untested edge cases.
- pdb++ recommended for debugging. CodeSignal's graphical debugger has "multiple severe issues" — don't use it. Print statements also fine.

**Test surface area (per email):**
- Python: classes/methods, list comprehensions, recursion, `collections.NamedTuple`, `unittest`
- Floats and floating point behavior
- NumPy: `np.array`, `np.min`, `np.max`, `np.sum`, `np.bincount`, `np.nonzero`, `np.where`, `np.random.RandomState`, array broadcasting

**Prep plan (5 days):**
- Take CodeSignal practice assessment to re-familiarize UI before attempting.
- Drill the listed NumPy functions — especially `bincount`, `nonzero`, `where` (less-used than min/max/sum) and broadcasting edge cases.
- `pdb++` install + tutorial if not already comfortable. Tutorial linked in email.
- Re-review LASR CodeSignal experience (2026-04-22): got stuck on numpy shape bug. **Direct lesson: shape-debug under time pressure is the failure mode to drill.** See [python-practice-plan.md](python-practice-plan.md).

**Day-of:**
- Stable internet, webcam, mic, photo ID ready.
- Open CodeSignal first; close Claude Code, Cursor, Copilot, any AI-assist editor. No external editor at all.

---

## Workstream + Mentor Notes (from claude.ai handoff, 2026-04-22)

**Workstream confirmed:** AI Safety Fellows, mechanistic interpretability focus.

**Likely mentors to read before research brainstorm interview:**
- **Trenton Bricken** (MTS): Towards Monosemanticity, Scaling Monosemanticity (SAEs on Claude 3 Sonnet), "On the Biology of a Large Language Model" (attribution graphs, transformer-circuits.pub, 2025), alignment auditing agents (2025). MATS Summer 2026 mentor. https://www.trentonbricken.com/
- **Samuel Marks** (MTS): leads cognitive oversight subteam of alignment science. Detecting deception/lying in LLMs, white-box + black-box techniques. Mech Interp Workshop NeurIPS 2025. MATS mentor. https://www.cbai.ai/samuel-marks
- Attribution graphs paper: https://transformer-circuits.pub/2025/attribution-graphs/biology.html

**Open thread:** claude.ai conversation ended with an unanswered question about Peter's background — revisit when picking back up.

---

## Still To Draft

- **AI safety background** (optional, 1 paragraph) — blog posts, IOI validation, arXiv preprint
- **Likelihood of full-time offer acceptance** — good chance; small chance academic pivot
- **Likelihood of continuing AI safety work** — almost certainly yes
- **Reference context** — background + relationship writeups for each coauthor

---

## Key Technical Facts (verified from literature)

- CLT attribution graphs are **per-prompt** — explicitly per-prompt in methods paper. CLTs themselves are corpus-trained.
- SAEs are corpus-trained (on the Pile). Features then applied per-activation.
- Methods paper explicitly states: "graphs do not contain information about influence on attention patterns" — QK circuits excluded.
- Peter's approach is **across-prompt** and **intervention-free** — complementary, not competing.
- Induction head formation: sharp phase transition, predicted by burstiness + diversity + repetition. Singh et al. 2024, Reddy 2024, Kawata et al. 2025 are key refs.

---

## Two-Essay Structure

The essays work as a diptych:
- **"Excited area"** = intellectual case: the science, why it matters for safety, the open question
- **"Why Fellows"** = personal case: what my background brings, what I've already found, what I need

Keep them non-redundant — technical content lives in "Excited area," trajectory/motivation lives in "Why Fellows."

---

## ⭐️ REFERENCES — reusable across apps (submitted verbatim to Anthropic Fellows 2026-03-30)

Ready-made "reference background + how we worked together" blurbs. **⚠️ STALE LINE:** the Palmer relationship text below says "I currently work as a researcher for her" — TRUE as of March 2026 but Peter's contract **ended end of June 2026**, so for any app after that, drop/past-tense it (e.g. "I have worked with her since January 2021, most recently as a researcher in her lab"). See MEMORY.md Current Work / bridge-income.

**Reference 1 — Stephanie Palmer** (stephanie.e.palmer@gmail.com)
- *Background:* Associate Professor, Dept. of Physics & Dept. of Organismal Biology and Anatomy, University of Chicago. Google Scholar: https://scholar.google.com/citations?user=0gtvj54AAAAJ&hl — Recently named a Schmidt Science Polymath: https://www.schmidtsciences.org/six-professors-named-schmidt-science-polymaths/
- *Relationship:* Stephanie was my PhD advisor. I currently work as a researcher for her [⚠️ now past-tense — see above]. We have worked together since January 2021. Our main work has included an ICML workshop paper and arXiv preprint under review at Physical Review Research. arxiv.org/abs/2512.09152 · https://openreview.net/forum?id=vxn5QGPFyi#all

**Reference 2 — David Schwab** (davidjschwab@gmail.com)
- *Background:* Associate Professor, Biology & Physics, CUNY Graduate Center. https://scholar.google.com/citations?user=UreckXYAAAAJ&hl=en · Member of technical staff at DatologyAI
- *Relationship:* One of my main collaborators; we worked on the above papers (arXiv preprint + ICML workshop). Worked together since late-2021.

**Reference 3 — Vudtiwat Ngampruetikorn ("Wave")** (wave.ngamp@gmail.com)
- *Background:* Senior Lecturer (Assistant Professor), School of Physics, University of Sydney. https://wavengampruetikorn.github.io/ · scholar.google.com/citations?user=cREUeIYAAAAJ&hl=en
- *Relationship:* Another main collaborator; worked on the above papers. Started when he was a postdoc for David Schwab. David and Stephanie gave higher-level feedback; Wave helped on finer details of simulations, coding, math, etc.
