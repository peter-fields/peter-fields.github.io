---
name: singapore-aisf-app
description: "Singapore AI Safety Fellowship (SASH) 2026 application — SUBMITTED 2026-07-10. Final submitted answers (why-fellowship, research-areas, collaboration), reference blurb, mentor picks, program facts, pipeline."
metadata: 
  node_type: memory
  type: project
  originSessionId: d7a4de9a-c915-4ce7-bfe9-07fa820e27cf
  modified: 2026-08-03T14:46:19.600Z
---

# Singapore AI Safety Fellowship (SASH) 2026 — Application

**Status:** ✅ **SUBMITTED 2026-07-10** (deadline day).
**Program:** independent, full-time **residential** research fellowship, Singapore (office in Chinatown). **Sept 21 – Dec 4, 2026** (~10.5 wks). Bridges technical AI safety with policy; East↔West cross-regional collaboration is a core pillar.
**Package:** SGD 5,000/mo stipend + fully-funded accommodation + travel assistance + **up to USD 30K compute** per eligible project + weekly mentorship + career dev.
**Advisors:** Zhang Ya-Qin (Tsinghua), **Ryan Kidd (MATS co-Exec-Director)**.
**Visa:** Peter is a US citizen → visa-free tourist *entry*, but the paid residency needs a **work/training pass the PROGRAM sponsors** (answered "yes, requires sponsorship"). See [[index-apps-and-projects]].

---

## Pipeline (⚠️ note stage 2)
1. ✅ Written application (~1 hr) — SUBMITTED 7/10
2. 🟢 **SHORTLISTED → technical work test invited 2026-07-31** (see below)
3. Mentor-specific work task and/or interview
4. Final selection

---

## ✅ Technical work test — invited 2026-07-31, **SUBMITTED (confirmed 2026-08-03)**
**Post-mortem so far:** one question tracked food items with a `defaultdict`, then extended to **expiration dates + sub-batches** — the part Peter was unsure how to structure in the moment. Worked-out pattern written up afterwards in `notebooks/practice/singapore_test_prep/batches_pattern.py` (runnable): sub-batches = change the *value type* (`defaultdict(int)` name→qty becomes `defaultdict(list)` name→`[Batch(expires, quantity)]` kept sorted by expiry), FEFO removal spilling across batches via `min(b.quantity, remaining)`, validate-before-mutate, prune emptied batches *and* the key. ⚠️ Trap that likely made it feel awkward: **a bare read `d[key]` on a defaultdict CREATES the key** — use `.get()` for reads.
**Next stage if it lands:** mentor-specific work task and/or interview.
**Format:** **Coderbyte** (link `coderbyte.com/sl-candidate?inviteKey=zrKIQG7KOe`), **~1h30m, proctored.** A **staged Python task: implement an inventory management system**, with a **provided test suite** to check your work. Graded on "problem-solving approach, coding ability, **attention to detail**."
**Rules:** no external resources, **no AI in any capacity (= disqualification)**, no external IDE (type into the Coderbyte IDE), browser search allowed in the lower-left panel *for syntax only*. Issues → fellowship@aisafety.sg.
**⚠️ Integrity line (same as [[arena-test-prep]]):** CC = **PREP ONLY**, zero help during the test.
**Prep materials (built 7/31, ~1 hr before the attempt):** `notebooks/practice/singapore_test_prep/` — `PREP_BRIEF.md` (format/grading read, first-5-minutes routine, staged working rhythm, trap list, inventory-shaped Python cheat sheet) + `mock/` (a 40-min timed mock: **library lending system**, deliberately a different domain, 34 stdlib-`unittest` tests in 4 stages + `SOLUTION_do_not_open.py`). Note: **no pytest installed anywhere on this machine** (base/py311/sca/inducing-induction) → mock uses `unittest`; run `python -m unittest -v test_library`.
**Core lesson to carry in:** validate *before* mutating (failed op must leave state unchanged); read the whole test file first; run the FULL suite after every stage so stage 3 doesn't break stage 1.

---

## Mentor picks (choose 3 max) — ✅ FINAL SELECTION (confirmed by Peter 2026-07-10)
Peter selected: **Krampis, James Chua (Anthropic), Min-Yen Kan.** (Took CC's Krampis #1 + Kan, but swapped Tan Zhi Xuan → Chua for the Anthropic-network play.)
1. **Konstantinos Krampis** (Hunter/CUNY) — ★ strongest fit. Topics: core mech interp + **new SAE-precision methods**, persona emergence, **bio foundation models/genomics**. Only mentor bridging Peter's protein/biophysics PhD *and* interp. Explicitly open to LessWrong "how can interpretability researchers help AGI go well" topics → Peter's QK/superposition ideas fit.
2. **James Chua** (Anthropic MTS) — 2nd proposed topic is "improving explainability of LLMs through training methods and **interpretability**" (his 1st is CBRN benchmarks = not Peter). Chosen mainly for the **Anthropic-network value** (Peter's target org, told to reapply ~2027); lower research-fit than Krampis/Kan.
3. **Min-Yen Kan** (NUS) — lists "AI Interpretability" first + safety benchmarks; at host institution (matching + network bonus).
- Not picked (CC's #3 rec): **Tan Zhi Xuan** (NUS) — model-based/rational AI, cognitive science, alignment foundations; closest to Peter's "physics of emergence" identity if a future cycle reopens.

---

## FINAL SUBMITTED ANSWERS

### Q1 — "What drew you to this fellowship? What are you hoping to gain?" (~300w)
> My background in statistical physics and information theory (with applications in biology/neuroscience) makes me well-positioned for frontier AI safety research; this fellowship program would actualize my potential. The prospect of applying my physics mindset to interpretability of AI excites me greatly. This is what I am looking for: working with others who passionately pursue a fundamental understanding of AI, but do so with an eye towards safe, effective deployment of AI—a cause I recently became more passionate about through BlueDot Impact's Technical AI Safety Course. Furthermore, it would be a privilege to bolster international collaboration on a research area so important for humanity's future safety and flourishing.
>
> My working theory is that interpretability is the most reasonable path forward in AI safety. Understanding how AI works (or at least the limits of that understanding) seems necessary for assessing risk. This is often thought to be in tension with behavioral approaches to AI research. One of my goals is to gain perspective on how interpretability research could complement behavioral research/interventions.
>
> I have seen direct evidence that my background is well suited for AI safety research: in recent blog posts (peter-fields.github.io), I ported concepts from theoretical neuroscience to circuit analysis in mechanistic interpretability. I tracked statistics of attention heads over different kinds of prompt classes (like tracking neural population statistics over different visual stimuli) in order to see which heads change their behavior under varying conditions and which do not (like revealing stimulus-independent info-processing structure in the retina). I developed characterizations of attention distributions from ideas in info theory and statistical physics, and found statistically significant signatures that separated circuit attention heads from non-circuit attention heads for GPT-2 small's Indirect Object Identification circuit (p<0.001).

### Q2 — Technical work sample (upload)
temp-tune / arXiv 2512.09152 (temperature-tuning paper) — same pick as PrincInt/BlueDot.

### Q4 — "Research areas + how prior experience helps" (~300w)
> I am excited to conduct research on the geometry of activation space in the residual stream, and its implications for computation.
>
> Many current lines of research in LLMs consider the residual stream vector space as Euclidean (e.g. persona vectors [arXiv:2507.21509] or activation plateaus [lesswrong.com/posts/WMfSbt7AAcJdHzysB]). However, the model never uses that inner product to compare residual stream vectors. The geometry it actually uses is implicit in the QK matrices of each attention head. Might the symmetric part be thought of as a metric? An interesting preliminary result from current work in progress: in GPT-2 small, the antisymmetric parts of the QK matrices have little overlap with eigenvectors of the token embedding matrix, suggesting a notion of a "compute-only" subspace in the residual stream used to transfer information among tokens.
>
> This prompts several possible directions of research. A next step is to check if B's top modes are orthogonal to activation space directions defined by SAEs. Metric-aware decomposition of QK matrices may further uncover a compute subspace that is mission critical for LLM computations but that SAEs are blind to. Furthermore, it would be interesting to see how these non-content directions are acted upon by MLP computations. If behaviors like deception utilize this subspace, uncovering such a mechanism would be a large step towards safe monitoring of AI systems.
>
> My background in physics and biology is well-suited for these questions. I have a unique blend of experience that utilizes observational techniques to uncover sparse structure in biological systems—as well as the ability to develop simple toy models that elucidate mechanisms and pathologies in learned generative models. More broadly, my training as a physicist helps me to think in terms of fundamentals—designing and conducting experiments end-to-end and drawing conclusions with implications for real-world applications from idealized scenarios/experiments.

*(Q4 sourcing: QK-geometry from Sharkey REAL-FINAL; metric-question + compute-only preliminary from PrincInt; SAE direction from Sharkey ORIG. The MLP-interaction clause is NEW here — the honest framing Peter settled on: the compute subspace is not *where* superposition happens; the interesting Q is how MLPs act on these non-content directions. B is a flashlight (reveals a non-content residual-stream region), not a fence — MLPs read the whole stream. See [[idea_qk_metric]], [[mats-sharkey-proposal-notes]].)*

### Q5 — "A time you worked with people who approached a problem very differently. How did you build alignment?" (~300w)
Story = his **ICML workshop paper "Understanding Energy-Based Modeling of Proteins via an Empirically Motivated Minimal Ground Truth Model"** with **Rama Ranganathan** (experimental biologist, protein science). Difference: biologists reason empirically from real proteins; Peter (physicist) builds idealized toy models. Built alignment by **embedding in Ranganathan's group** — attending group meetings, reading his papers, talking with students — until fluent enough in their framing to build a toy model they'd trust. (Full text in submitted app; not repeated here — reconstructable from these beats.) *No prior app had a ready-made answer for this Q — written fresh this cycle.*

### Reference — Stephanie Palmer (as submitted, single combined field)
> Stephanie is a theoretical neuroscientist and biophysicist at the University of Chicago and was my PhD advisor. We have worked together since January 2021. Our main work has included an ICML workshop paper and arXiv preprint under review at Physical Review Research. (arxiv.org/abs/2512.09152, https://openreview.net/forum?id=vxn5QGPFyi#all) https://scholar.google.com/citations?user=0gtvj54AAAAJ&hl Recently named Schmidt Science Polymath: https://www.schmidtsciences.org/six-professors-named-schmidt-science-polymaths/

*(Peter correctly dropped the now-stale "I currently work as a researcher for her" — contract ended end of June 2026. Full reusable reference blurbs for all 3 coauthors, incl. Schwab + Ngampruetikorn, live in [[anthropic-fellows-app]].)*

---

## Cross-app reuse map (what came from where)
- Q1: Anthropic-Fellows "Why Fellows" + PrincInt "exactly what I'm looking for" + BlueDot landscape framing. Peter added the international-collaboration sentence fresh (only lightly endorsed the East/West + policy pillars — "safe and effective deployment" covers policy for him).
- Q4: [[mats-sharkey-proposal-notes]] + [[princint-app]] (the SAE-blindness sentence traces to Sharkey ORIG, self-written but never form-submitted).
- Q5: fresh; only seed was the Palmer-lab cross-disciplinary line.
- Reference: [[anthropic-fellows-app]] reference section.
