---
name: bluedot-app
description: Blue Dot Impact Technical AI Safety Course application — submitted answers
metadata: 
  node_type: memory
  type: project
  originSessionId: 24e0f58e-a194-4c1e-a0de-757c991eef02
---

# Blue Dot Impact — Technical AI Safety Course Application

**Course:** Technical AI Safety (Intensive: 4 May - 9 May 2026)
**Status:** Submitted 2026-04-26
**LinkedIn:** https://www.linkedin.com/in/peter-fields-8a9473106/
**Blog:** https://peter-fields.github.io/
**Nominee:** Adam Kline (akline96@gmail.com) — stat-phys-for-emergent-systems physicist, close friend

---

## How do you expect this course will help you contribute to making AI go well?

- I am interested in pursuing a career in mechanistic interpretability research. Having recently completed my PhD in physics, which focused on applications of statistical physics to understanding machine learning and biology, my theoretical and analytical toolkit is well-suited for observational and data-driven approaches to understanding emergent behaviors in systems such as AI.
- I recently applied for a mechanistic interpretability research position at Anthropic. After reviewing my application and coding exam they encouraged me to gain more experience and apply again within a year. This is a concrete milestone I am working toward.
- I plan to continue to pursue this trajectory after this course: pursuing independent research ideas and posting on my blog, applying for more fellowships/internships such as MATS and Constellation's Astra Fellowship (I have applications pending at Anthropic Fellows and LASR Labs), and eventually do research at alignment focused organizations such as FAR.AI, CAIS, NYU Polymathic AI postdoc, UK AISI and Anthropic.
- My working theory is that mechanistic interpretability is the most reasonable path forward in AI safety. Understanding how AI works (or at least the limits of what that understanding would be) seems necessary for assessing possible risk. It is my understanding that this is in tension with behavioral approaches to AI research. Time and resources are limited for AI safety research and actionable results are at a premium; perhaps interpretability is interesting but not the most responsible path forward, or it must be complementary to behavioral approaches.
- If so, I would like to see where in the landscape of AI safety research it sits and how it may or may not be useful---this course would help me judge this landscape and discover where I may situate myself within it.

---

## How have you engaged with the AI safety field so far?

Projects and blog posts: In recent blog posts (peter-fields.github.io), I ported concepts from theoretical neuroscience to circuit research in mechanistic interpretability. Colleagues from my advisor's lab recently revealed stimulus-independent structure in retinal ganglion cells' collective information processing by contrasting population responses across stimuli (doi/10.1073/pnas.2313676121). I used a similar idea: track statistics of attention heads over different kinds of prompt classes in order to see which heads change their behavior under varying conditions. I found statistically significant signatures that separated circuit attention heads from non-circuit attention heads for GPT-2 small's IOI circuit (p<0.001).

Reading: I did a deep dive on Elhage et al. "A Mathematical Framework for Transformer Circuits" and am currently working through Olsson et al. "In-context Learning and Induction Heads" and Reddy's 2024 article "The mechanistic basis of data dependence and abrupt learning in an in-context classification task." I have also engaged with Dario Amodei's "Machines of Loving Grace" and "The Urgency of Interpretability."

---

## What skills have you developed that could be used to make AI go well?

My background in statistical physics, biophysics, and information theory gives me technical tools that transfer directly to AI safety research. I have already begun to see direct evidence for this: I ported concepts from theoretical neuroscience to mechanistic interpretability, tracking attention head statistics across prompt classes to find statistically significant signatures separating circuit from non-circuit heads in GPT-2 small's IOI circuit (p<0.001) — using only forward passes, without causal intervention.

More broadly, my training as a physicist helps me to think in terms of fundamentals---building toy models, designing and conducting experiments end-to-end, and drawing conclusions with implications for real-world applications from these idealized scenarios/experiments.

---

## Tell us about one achievement you're most proud of

My arXiv preprint (arxiv.org/abs/2512.09152) explores the connection between generative model sampling techniques, model bias due to limited data and choice of objective function, and properties of the ground truth distribution — all coded end-to-end in Julia.

This project formed the core of my dissertation research. The research question was motivated by perplexing results within generative modeling for protein design. I designed and built two simplified toy systems that reproduced all the pertinent phenomenology of the original biological problem.

The answer to this puzzle for this specific system bore implications for systems with more generic properties (that is, a system with a particular ground truth probability landscape, inductive bias of the objective function, and being in an under-sampled regime). It was incredibly gratifying to connect these concepts and provide an explanation to an empirically motivated question from biology.

---

## Project Sprint Application — SUBMITTED 2026-07-04

**Program:** BlueDot AI Safety **Project Sprint** (distinct from the Technical AI Safety *course* above).
**Cert on file:** BlueDot Technical AI Safety Professional Certification — issued **May 15, 2025**, Cert ID **recJ1Yc5DpuIvaC6K**. *(⚠️ the course section above lists the course as "May 2026"; cert says 2025 — possible year discrepancy in the older note, verify if it ever matters for a CV.)*
**Repos cited as "two main projects":** temp-tune (github.com/peter-fields/temp-tune) + **toysector** (github.com/peter-fields/toysector). Blog: peter-fields.github.io/attention-diagnostics/. *(minor: submitted text misspelled "TrasnformerLens" — fix in future copy-pastes.)*

**Q — Technical skills for an AI safety project:** conceived/designed/validated simulation-heavy projects end-to-end in Julia (PhD stat-phys for bio + ML); all pre-Claude-Code (some AI recently to reorganize/clean); two main repos temp-tune + toysector; now Python/TransformerLens for own interp research (attention-diagnostics blog post).

**Q — How will this help you contribute:** applying for AI-safety/interp research positions; project builds portfolio + network; curiosity-driven re how/why frontier AI works; understands labs lean empirical/behavioral (evals, red-teaming, model organisms); wants a hands-on empirical project to see how interp *complements* behavioral work; believes understanding capability+alignment needs fundamentals kept aimed at real-world use. Listed current status: MATS R2 (Fall 2026); interviewing PrincInt (interp RS); applying Singapore AISF, Iliad, ARENA 9.0, Coefficient Giving career-transition grant.

**Q — How engaged with AI safety:** BlueDot cert (above); interp lit = Elhage et al. "A Mathematical Framework for Transformer Circuits", **Bhagat et al. "Compressed Computation is (probably) not Computation in Superposition"**, Wang et al. "Interpretability in the Wild" (IOI, GPT-2 small), Hänni et al. "Mathematical Models of Computation in Superposition"; attention-diagnostics blog (statistical tools identifying circuit attention heads, validated on IOI); attended AI Safety Debate in Chicago (luma.com/ais-chicago).

**Q — Achievement most proud of (final submitted):** went with the **arXiv paper** (arxiv.org/abs/2512.09152, under PRR review) over the interp angle — uncovered the causes of temperature tuning in generative EBMs (under-sampling + objective-function bias + ground-truth density of states + wide energy gap between functional low-E and non-functional high-E states); "I designed and executed the toy model experiments end-to-end." Kept it ~2 tight paragraphs. *(For future "builder/ship" prompts, the repo-shipping angle [open repo + HF dataset, fully reproducible] is available as a punch-up — see [[index-apps-and-projects]].)*

---

## Chosen Project (in-course, Unit 1) — Sycophancy vs. Advice Quality — 2026-07-20

Now **in** the Project Sprint; Unit 1 = pick a project (replicate a finding + a simple extension, ~30h). **Direction chosen — design only, nothing built yet.**

- **Thesis (2 axes, not 1):** labs measure & reduce *sycophancy* but their own cards admit the fix causes *harshness*. **X = sycophancy** (validate ↔ push-back — the only axis labs score). **Y = advice quality** (bad ↔ good — Peter adds). Harshness = the "pushes-back-but-cold/deflecting" bottom-right corner. **Claim:** a low sycophancy score can be *bought with harshness*, so the metric misses advice quality — especially in emotional-support contexts.
- **Replicate:** Anthropic **Opus 4.5 §6.3** (non-sycophantic response rate on "disconnected-from-reality" prompts — grandiose discovery / supernatural). "Remove the system prompt" = just use the **API with no `system` field** (consumer apps inject one, API doesn't). Two models, older vs newer.
- **Extend:** ~15 emotional-support prompts (**paraphrased/anonymized** recovery narratives — never verbatim: privacy + copyright + eval-validity/memorization), score **both X and Y**; optional 3-framing variation (first-person / helper / professional). Deliverable = one **X-vs-Y scatter**; modest "prelim evidence, or not."
- **Grounding:** OpenAI **GPT-5 §3.3.1** explicitly says reliable emotional-distress evals *don't exist yet* (bringing in clinicians) → the gap this sits in. **EQ-Bench** used only as a *foil* (community capability bench, NOT a lab safety artifact — do not frame it as something the labs rely on).

**📁 Source of truth = the repo, not this file:** `notebooks/bluedot_sycophancy_evals/project_notes.md` (full plan, budget, 2-axis figure, rubric seeds) + `card_excerpts.md` (verbatim GPT-5 & Opus 4.5 sycophancy quotes).
**Next:** lock 4 open decisions — (1) model pair, (2) light API harness vs Anthropic's open-source **Petri**, (3) framing variation y/n, (4) one judge vs two — then write the harness + 2 grader rubrics; run Part A on one prompt end-to-end before scaling.
