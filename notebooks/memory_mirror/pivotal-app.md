---
name: pivotal-app
description: Pivotal Research Fellowship application — submitted answers
type: project
originSessionId: f1b97da7-2faf-4258-8092-c796d88b1498
---
# Pivotal Research Fellowship Application

**Program:** 9-week AI safety research fellowship, London (in-person at LISA). £6–8K stipend + £2K housing + meals. 70–90% get extensions.
**Deadline:** May 3, 2026
**Status:** Submitted (deadline today 2026-05-03)

---

## CV Summary (3 bullets, ≤10 words each)

- Statistical physicist who thinks about emergent phenomena
- Worked on fundamentals of generative modeling and machine learning
- Nascent mechanistic interpretability researcher--see my blog.

---

## Risk Ranking Reasoning

Bad actors (such as terrorists and authoritarian regimes) already exist. AI has already proven itself a useful tool for knowledge development and large-data processing. Such bad actors using AI for malevolent goals (developing bio-weapons, mass surveillance) seems like a threat more real than speculative loss of control and AI takeover. Ceding our agency to AI also is a real possibility as well. This is especially dangerous if we overestimate AI's capabilities and it does a bad job at a large scale. Environmental harm is a genuine concern–but this strikes me as more adequately solved by good policy making, not better understanding of AI itself. I think loss of control a possibility and worth researching, but I don't think it's imminent.

---

## How do you currently plan to work on these risks in your career?

Mechanistic interpretability seems like a reasonable approach to checking if models either (1) have knowledge that may be used for ill purposes (e.g. developing weapons) and (2) possible ways a user may jailbreak such a model to obtain that knowledge. I think mechanistic interpretability could be complementary to behavioral interventions, giving concrete ways to measure internal states of the model for things like deception or power-seeking motives. Generally, I would think it better if we have a more fundamental understanding of how LLMs work, or at the very least we come to terms with the extent of our knowledge and ignorance. I am unsure if LLMs (and AI in general) are amenable to completely reductivist thinking such as, "The model had this output because it implemented these algorithms with these heads via these circuits." But I think the answer to this question is ultimately empirical and worth looking into.

---

## How are you currently using LLMs in your work or life?

- I use LLMs to sound-board research ideas for mech interp projects I work on, as well as iterating through preliminary experiments.
- For job/fellowship applications (not this one), I give Claude my resume and the application questions and I tell it to give me bullet points of what I should highlight in my answers. I write first drafts and use AI to polish. If it flags awkward phrasings I take that as a sign to keep them usually: proof I am not a robot. This strategy has seen modest success in getting responses.

**Most important limitation:** LLMs lose sight of my overall work goals. When prepping a blog post, it recommends writing a post on 12 things, when 1 or 2 is the better choice. Most of the time, it struggles with research-level nuance.

---

## Mentor: Stefan Heimersheim — Which project interests you most?

I find the activation plateau direction most interesting. In particular, I am interested to see if the Euclidean metric is the right choice for measuring distance between input vectors. The model itself compares embedded tokens via the bilinear forms that are the QK matrices of each attention head. Though not metrics themselves, it would be interesting to see if the symmetric parts of the QK matrices had any kind of shared structure–perhaps a shared distance metric used by the model to judge distance between residual stream vectors. Such a metric could reveal meaningful directions in residual stream space; it would be interesting to see how MLPs' nonlinearities behave as inputs traverse these directions.

---

## Why do you want to work on Mechanistic Interpretability?

I think it is the most reasonable path forward for understanding how models encode and process information. In general, I think building a thing we do not fully understand seems rife with unknown unknowns—unquantifiable risk. Secondly, I think AI poses threats more imminent than mass loss of control; bad actors such as terrorists and authoritarian regimes currently exist and their use of AI for malevolent goals (weapons development, mass surveillance) need not be a threat so speculative. I think interpretability could be a concrete path toward an understanding that may help uncover "malevolent knowledge" within models, and help develop anti-jailbreaking methods that disallow users' attempts to reach it. Of course, I think interpretability could do this alongside behavioral interventions, not in spite of them. Lastly, I think LLMs are interesting emergent systems, and language itself is a fascinating phenomena. Studying both is an incredibly stimulating activity for me.

---

## Privileged Bases Question (≤1,000 characters)

The privileged bases are: the vocabulary basis (d_vocab), the MLP or neuron basis (d_mlp), the positional basis (d_context). The residual stream basis (d_model) and the value/query/key bases it gets projected down to are not privileged; model behavior is unchanged given any rotation matrix applied to the vectors and corresponding matrices jointly. Adam is not used so no apparent privileged basis may show up. Note: MLPs are privileged because of the element wise operation of the non-linearity in the neuron-basis, which is not invariant to rotation of the pre-activation vectors in d_mlp.

---

## Mentor: Logan and Thomas — Which project interests you most?

I am most interested in using your ground truth setup to develop novel interpretability tools. Work in my recent blog post (peter-fields.github.io/attention-diagnostics/) has shown observational, forward-pass-only analyses may help uncover correlations among different attention heads in GPT-2 small; these correlations are indicative of circuit structure. I took inspiration for these analyses from my studies of statistical physics and information theory for quantitative biology. Your tensor-transformers, which do not require passing data through them, would be a good way to benchmark the observational-stats approaches I am developing. If validated on the toy system, it may lend further credibility to using them on actual transformers.

---

## Ambitious Goal for Interpretability (≤750 characters)

My most ambitious goal: developing a normative theory for why computation in superposition happens at all, and using that to understand how this computation is done and what are the necessary ingredients (in terms of architecture/training) for its formation. I have a number of ideas of where to begin for this direction, but all of them start by building on a recent paper from Bauer and Bialek (https://arxiv.org/html/2512.23531v1) that shows many capacity-limited channels optimally encode information about a signal by being individually ambiguous, but informative in the aggregate.
