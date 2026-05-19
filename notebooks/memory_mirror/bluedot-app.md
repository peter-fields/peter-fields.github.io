---
name: bluedot-app
description: Blue Dot Impact Technical AI Safety Course application — submitted answers
type: project
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
