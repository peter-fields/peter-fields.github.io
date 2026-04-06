---
name: anthropic-fellows-app
description: Anthropic Fellows Program (July 2026) application — settled essay bullets and key framing decisions
type: project
---

# Anthropic Fellows Program — Application Notes

**Program:** July 2026 cohort, 4 months, paid ($3,850/week), Berkeley workspace
**Form:** https://airtable.com/appiuxxfhf5moRwTx/pagULi7KjbpUaOdIg/form
**Status:** IN PROGRESS — essays drafted as bullets, not yet written as prose

---

## Key Framing Decisions

- **Do NOT mention the RS Interpretability rejection in the essays.** Form surfaces previous application; let it. Don't lead with it.
- **Don't oversell a specific research agenda.** Pitch is: "I've done real work, here's one direction, I'm here to learn and contribute to whatever the team thinks is highest priority."
- **Code sample:** temp tuning repo only. IOI notebook was Claude-generated (Peter verified line-by-line but don't submit as code sample).
- **References:** coauthors on temp tuning paper (Schwab, Ngampruetikorn, Palmer). Pick whoever can speak most concretely to independent contributions.
- **Full-time offer:** good chance of accepting; small chance of academic pivot if AI research feels like wrong fit.

---

## Essay 1: "Why Fellows?" (1-2 paragraphs)

**Para 1 — what I bring:**
- Confident my background has real contributions to make — not because the problems are easy, but because I've already transferred a specific method directly
- Prompt-contrast idea came from lab work: varying stimuli to reveal stimulus-independent structure in retinal ganglion cell populations — circuit shows up in the contrast, not in any individual stimulus. Same logic applied to attention heads across prompt classes.
- Already got results: statistically significant discrimination of circuit vs non-circuit heads (p<0.001), forward-pass-only, no training, no intervention
- Anthropic researchers found it interesting enough to advance me to interview

**Para 2 — what I need + closer:**
- What I'm missing is context only available from inside: what's been tried, what matters, where my instincts are right and where I'm reinventing the wheel
- Not looking to be handed a project — want mentors who help me find my footing fast in a new field
- Confident that combination — novel background, preliminary results, embedded collaboration — leads to genuine contributions
- **Closer:** end with the RS rejection quote — "we thought your application and [coding] test results were promising, and we would like to encourage you to consider applying again in a year after you've gained more experience." This program is exactly that. The answer to "why this fellowship?" is: because I'm following through.

---

## Essay 2: "Excited area of AI safety?" (1 paragraph)

- Mechanistic interpretability: understanding internal computational structure, not just input-output behavior
- The prize I keep thinking about: if you can connect circuit formation to training data statistics, you can predict capability emergence before it happens — directly safety-relevant
- Induction heads are proof of concept — formation predicted by burstiness, diversity, repetition in training data. Does this generalize to more complex circuits?
- Map/territory: what a circuit computes and what data statistics produced it are two sides of the same reverse-engineering problem — each constrains the other
- Current interp work is mostly post-hoc and diagnostic; want to push toward predictive
- Prompt-class contrast is one empirical handle: probe which data statistics a circuit is sensitive to by varying what you show it

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
