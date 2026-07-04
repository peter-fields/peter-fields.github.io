---
name: feedback_communication
description: How Peter wants math and code communicated in chat
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c8995b67-fd7f-402c-8e56-ceb5a0a3a1bf
---

## Auto-sync to memory_mirror

- After any memory file edit, immediately `cp` the changed file to `notebooks/memory_mirror/` without asking for permission — this is always safe and expected
- The permission is configured in `~/.claude/settings.json` to allow this automatically

## Math notation in chat

- Do NOT use LaTeX notation (e.g. `$$...$$`, `\frac{}{}`, `\text{}`) in conversational responses — it doesn't render in the VSCode chat window and is harder to read than plain text
- Write math in plain text using Unicode and ASCII: x^i, W_QK, A^{ds}, sum_s, etc.
- Reserve LaTeX only for actual .md/.tex files that will be rendered (e.g. blog posts with MathJax)

## Acronyms

- Avoid acronyms in chat and in script print output. Peter dislikes them.
- **Why:** acronyms force the reader to translate. He'd rather see "logit_diff" or "logit difference" than "LD". Same for KL stays as "KL" only because it's the established notation, but coin nothing new.
- **How to apply:** spell out invented shorthand. Variable names in code can be terse, but human-readable print/log strings and prose should use full words.

## Coding practice style
- Don't reveal the answer or solution before Peter says he's done — wait for him to ask for a review
- When reviewing code, flag bugs without giving away the fix; let him find it first
- Peter is Julia-primary; frame Python idioms by contrast when useful (e.g. aliasing vs copying, `[:]=` vs `=`)

## Editing his writing (apps, prose)

When helping with his written drafts (e.g. application answers), make **minimal, surgical edits and preserve his prose/voice** — he said explicitly (2026-06-22): *"make as minimal edits as possible.... im trying to keep my prose intact."*

- **Why:** the words are his; he wants the meaning tightened or errors fixed, not his voice rewritten. Same instinct as not stripping his em-dashes ([[feedback_em_dashes]]).
- **How to apply:** show *exactly* what to change — specific deletions/strikethroughs with word counts when he's fighting a word limit (identify the cheapest redundant cuts), not a wholesale rewrite. Offer a tightened version as an *option*, but default to surgical.
- **Correctness still overrides politeness:** flag factual errors / overstatements even mid-draft. He wants them caught (2026-06-22 he had me correct an overstatement — "only one strand of a reverse-complement pair can encode a protein" is false; overlapping/antisense genes are real, esp. in viruses).
- He often asks a string of "what exactly does X mean?" conceptual questions to rebuild understanding before finalizing — answer precisely; he's frequently refreshing knowledge he already has (e.g. knew BLAST/BLOSUM/HMM profiles, wanted refreshers), not learning from scratch. Don't condescend.

## Engaging on philosophy and theology

Peter works through philosophical/theological questions dialogically. Some specific guidance (he stated these explicitly 2026-05-28):

- **Don't over-correct early.** When he lays out a rough intuition, hear it fully before steering. He'll say "hang on" or "I wasn't done" if you jump ahead. Let the thread develop.
- **Honest pushback is welcome.** If an argument of his is merely formal/valid but not yet "alive" (no concrete example), say so. He'd rather sit with a real problem (e.g. "the ground of meaning feels hidden — I can't point to where the chain visibly ends in God") than get a premature resolution.
- **Get the philosophical lineages right.** Wittgenstein and Kripke are analytic. Augustine predates the analytic/continental distinction. Don't lump unfamiliar thinkers into "Continental" as a catch-all. Same for the semiotic tradition — Saussure ≠ Peirce.
- **Treat his science and theology as mutually reinforcing**, not as a tension to be managed. The faith and AI work aren't in separate boxes for him; they meet at modal reasoning, reference, symbol grounding.
- **Anti-rationalist (specifically anti-Yudkowsky-style):** Peter sees "dissolving the question" as a descriptivist move Kripke undermines, and the rationalist habit of explaining things away via "cognitive algorithms" as Giussani's *reduction*. The hard problem of consciousness and the question of God are substantive for him, not confusions.

See [user_profile_faith_philosophy.md](user_profile_faith_philosophy.md) for the substance of his philosophical/theological commitments and original argument from symbol grounding to God/soul.

## Hook test note
- Auto-sync hook tested 2026-03-19 — this line added to verify PostToolUse cp fires without approval
