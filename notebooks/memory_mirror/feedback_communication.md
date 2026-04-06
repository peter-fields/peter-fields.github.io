---
name: feedback_communication
description: How Peter wants math and code communicated in chat
type: feedback
---

## Auto-sync to memory_mirror

- After any memory file edit, immediately `cp` the changed file to `notebooks/memory_mirror/` without asking for permission — this is always safe and expected
- The permission is configured in `~/.claude/settings.json` to allow this automatically

## Math notation in chat

- Do NOT use LaTeX notation (e.g. `$$...$$`, `\frac{}{}`, `\text{}`) in conversational responses — it doesn't render in the VSCode chat window and is harder to read than plain text
- Write math in plain text using Unicode and ASCII: x^i, W_QK, A^{ds}, sum_s, etc.
- Reserve LaTeX only for actual .md/.tex files that will be rendered (e.g. blog posts with MathJax)

## Hook test note
- Auto-sync hook tested 2026-03-19 — this line added to verify PostToolUse cp fires without approval
