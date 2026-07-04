---
name: meta-fair-app
description: "Peter's Meta FAIR Postdoctoral Researcher (PhD) initial-prescreen answers — full Q&A, esp. the reusable AI-tools-usage answers"
metadata: 
  node_type: memory
  type: project
  originSessionId: e5ca9131-417c-4e30-8bad-b8194981fc8f
---

# Meta — Postdoctoral Researcher, Fundamental AI Research (FAIR), PhD

**Status:** Applied 2026-06-01; **initial prescreen SUBMITTED 2026-06-11.** Awaiting response (technical screen → full loop next). Postdoc on-ramp (not senior RS). Comp ~$122K–$181K. Values caveat: MSL is capabilities-first.
**Locations open:** Chicago IL, New York NY, Seattle WA, San Diego CA, Menlo Park CA, San Francisco CA, Santa Clara CA, Remote US.
**Internal contacts named:** Chuck Rossi; **David Schwab** (physicist/comp-neuro — natural stat-phys×ML connection).
**Resume:** uses a general resume (`notebooks/other_jobs/general_resumes/`, latest 2026-06-07).
**⚠️ Stale if reused:** the "other interviews" answer listed Anthropic Fellows (late July) + Astra/Constellation (Sept) as active — **both REJECTED 2026-06-11**, so update before reusing.

## Prescreen Q&A

**Why looking for new opportunities?** Recently completed PhD in Physics; dissertation on applications of statistical physics to machine learning and biological systems. Looking to continue this line of work in AI research.

**Most important things in next job?** A curiosity-driven research environment, experts in AI research to gain experience from, and an ability to meaningfully contribute to AI's ability to help humanity.

**Interview timing:** As soon as possible.

**(AI-tools Q1) Recent project using AI tools to achieve specific goals (3-5 sentences):**
> I use Claude Code to sound-board research ideas for mechanistic interpretability projects, as well as iterating through preliminary experiments. Recently, this helped me publish a blog post instantiating some of these ideas (peter-fields.github.io/attention-diagnostics/). I developed diagnostics for measuring per-prompt statistics of attention-head distributions, and wanted to see if one could track meaningful shifts in these distributions when different text prompts were fed into GPT-2-small. Claude Code helped me iterate through the development of prompts, track attention head statistics, and visualize results, eventually leading to a statistically significant identification of circuit versus non-circuit heads on the indirect-object (IOI) identification task.

**(AI-tools Q2) An issue you identified in AI output + how you corrected it:**
> While working on the blog post for distinguishing the IOI circuit-heads from non-circuit-heads in GPT-2-small using the shifts in attention distribution statistics under different prompts, initial results from AI-assisted experiments indicated no significant shift. However, I noticed that the text prompts were of varying length, and the control group varied in surface form rather than only IOI structure. I course corrected by redesigning the prompts, directed Claude to re-run experiments, and this led to clear shift in the attention distributions of circuit heads (p<0.001).

**(AI-tools Q3) How you keep up with AI + a recent thing learned/applied:**
> I am active on the Slack for BlueDot Impact, a non-profit focused on AI safety education, and took a short course with them on technical AI safety. I am still involved in the academic environment of University of Chicago, and regularly discuss AI technologies and innovations with other researchers. I follow a number of blogs on AI research and news, including Import AI, the Algorithmic Bridge, and Human Override. I read about AI research on LessWrong and keep up with Anthropic's mechanistic interpretability research. Most recently, I've been working through Hänni et al.'s 'Mathematical Models of Computation in Superposition' (arXiv:2408.05451) alongside Stefan Heimersheim's related work on compressed computation, and am sketching a project that extends the framework with information-theoretic arguments.

Related: [[compressed-computation-project]] (the Hänni/Heimersheim sketch mentioned in Q3), [[index-apps-and-projects]].
