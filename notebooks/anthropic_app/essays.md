---
title: anthropic app essay questions
---

## Why Anthropic? v3

I strongly agree with Dario Amodei’s assessment in 'The Urgency of Interpretability': the time to understand AI is before its advancement to a level beyond our understanding — that is, right now. How LLMs may generalize to contexts not explicitly trained on, the fundamental causes of deception and hallucination, jailbreaking methods yet unknown… all such questions and concerns are paramount, and can only be provisionally addressed until we understand how AI actually works.  

Anthropic has made the most progress on this front — from the superposition hypothesis through sparse autoencoders to circuit tracing and cross-layer transcoders. Because of Anthropic’s structure as a public benefit corporation, and its consequent resolve against commercial and political pressures, I trust it to protect its interpretability research program, and more generally, to value safety over unfettered progress. Many technological transitions have moved faster than our ability to understand them; I don't think AI should.

I have already begun contributing to interpretability research: developing forward-pass-only diagnostics for attention head circuitry and validating on the IOI circuit in GPT-2. My background in statistical physics has primed my intuition for the fundamental workings of many-component systems, from perspectives in information geometry to practical applications in real biological systems. Observational methods that inform costly interventions, and the tension between system-specific idiosyncrasy and universal structure — these are central to how I approach interpretability. 

Anthropic is where I can apply these skills to larger models, harder questions — and with a team that understands the scale of collaboration required.

---

<!--
### Why Anthropic? v2

In "The Urgency of Interpretability," Dario Amodei argues that the window for understanding AI systems closes as they become more capable — that this work has to happen now, before the gap between capability and comprehension becomes irreversible. I find this framing exactly right.

Anthropic's mechanistic interpretability program — from the superposition hypothesis through sparse autoencoders to circuit tracing and cross-layer transcoders — is the most advanced in the field. No other lab has invested comparably in the foundational question of what these models are actually doing. Anthropic's structure as a public benefit corporation, without the commercial pressures that have visibly bent other labs' research agendas, means this work is likely to stay protected.

I have already begun contributing to this program: developing forward-pass diagnostics for attention head circuitry and validating on the IOI circuit in GPT-2. My background in statistical physics — many-component systems, observational methods that inform costly interventions, the tension between system-specific idiosyncrasies and universal structure — transfers directly. Anthropic is where I can scale this up.
-->

---

## Why do you want to work on the Anthropic interpretability team? 
(We value this response highly - great answers are often 200-400 words. We like to see deeper and more specific engagement with Anthropic and our interpretability agenda than simply interest in AI or LLMs.)* 
### v3

Anthropic’s papers “Circuit Tracing” and “On the Biology of a Large Language Model” develop techniques that intervene at the level of the MLPs between attention layers. Cross-layer transcoders help build attribution graphs, tracing feature-to-feature computation through MLPs and OV circuits. But the QK circuits (Elhage et al., 2021) that determine which token positions attend to others are missing. Verifying QK circuit membership requires causal interventions that don’t scale to large models.

I have developed diagnostic tools for identifying candidate circuit heads. Rather than consider heads on a per-prompt basis, I track statistics of heads over several prompts of varying types. The first is the KL-divergence of a head’s attention distribution from uniform — a measure of selectivity. The second is the derivative of this quantity with respect to an auxiliary temperature parameter — a measure of susceptibility. Both require only a forward pass. In the IOI circuit on GPT-2, I have shown statistically significant shifts (p=0.0002 and p<0.0001) in circuit heads when switching between circuit-activating IOI prompts and similar yet non-activating non-IOI prompts.

This work was inspired by my work in statistical physics/machine learning for applications in biology, where observational methods avoid costly interventions while still revealing meaningful structure. This background informs my work in interpretability; covariance of my diagnostics between heads (and measured across prompts) has shown potential for a scalable method of culling heads down to fewer circuit candidates—as it avoids direct mechanistic intervention. 

I am also interested in the hypothesis that complex reasoning involves more distributed, less circuit-like computation — closer to reservoir dynamics — in which case my toolkit for collective dynamics is well-suited. And looking further: can inference time circuit diagnostic techniques be used to explicitly promote formation during training? One concrete direction: constraining attention selectivity during training, relaxing the constraint with layer depth; this idea follows directly from the information-theoretic framework I have been developing.

---

<!--
### Why do you want to work on the Anthropic interpretability team? v2

Anthropic’s papers "Circuit Tracing" and "On the Biology of a Large Language Model" develop techniques that intervene at the level of the MLPs between attention layers. Cross-layer transcoders build attribution graphs tracing feature-to-feature computation through MLPs and OV circuits — a major advance. But the QK circuits that determine which token positions attend to others are deliberately left out: a significant open gap. Per-prompt analysis of attention patterns is also computationally expensive and doesn’t scale.

My diagnostic tools are designed to address both problems. Rather than analyzing heads per-prompt, I track statistics over many prompts: the KL-divergence of a head’s attention distribution from uniform (selectivity) and its derivative with respect to an auxiliary temperature (stability). Both require only a forward pass. In the IOI circuit on GPT-2, shifts between circuit-activating and non-activating prompts are statistically significant (p=0.0002 and p<0.0001). Cross-head covariance of these statistics shows early promise as a scalable screening method — identifying circuit candidates before costly mechanistic intervention.

This follows from my background in statistical physics applied to biology, where observational methods do real work before expensive experiments. I want to develop this into a full pipeline at Anthropic, scaling from GPT-2 to frontier models. I am also interested in the hypothesis that complex reasoning tasks employ more distributed, less circuit-like computation — closer to reservoir dynamics — in which case my toolkit for collective dynamics is well-suited. And looking further: if circuits can be characterized at inference time, can we promote their formation during training? One concrete direction — constraining attention selectivity during training, relaxing the constraint with layer depth — follows directly from the information-theoretic framework I have been developing.
-->

---

## Please briefly describe the technical achievement you’re most proud of, including what you did, how you collaborated with others (if relevant), and its significance or impact.

My recent arXiv preprint (under review at Physical Review Research), “Understanding temperature tuning in energy-based models” is my proudest technical achievement. Energy-based models are used to generate novel protein sequences, ones that function in vivo. However, they typically employ a post-training temperature tuning of the learned energy function in order to get these functional sequences. My work gave this heuristic a principled explanation.

I did experiments on simple ground truth models and linked how finite sampling, standard objective functions, and the structure of real-world-data state spaces (functional states greatly outnumbered by nonfunctional states) create the bias that temperature tuning corrects. The analysis also indicated when it would be advantageous to raise instead of lower temperature. 

My advisor, Stephanie Palmer, and my main collaborators David Schwab and Wave Ngampruetikorn, saw the significance of my analysis before I did and pushed me to synthesize it into a coherent narrative, to understand its implications within the broader research community, and to take pride in work that I initially undersold (to myself most of all).  

To the best of my knowledge, there was not a theoretical framework within which to understand the separate yet related phenomena of finite-sample/objective function inductive biases, metrics of generative performance, ground truth data-generating processes, and generative modeling sampling tricks. By connecting these phenomena, my work allows practitioners of temperature tuning to understand why it is required in the first place and what underlying bias it corrects.
 
---

---

## Please share a link to the piece of work you've done that is most relevant to the interpretability team, along with a brief description of the work and its relevance.*

https://peter-fields.github.io/attention-diagnostics/

This post applies two diagnostics I developed to analyze the behavior of the IOI-circuit in GPT-2, implemented in Python using TransformerLens. The selectivity of the softmax distribution of a given head for a given prompt is measured via its Kullback-Leibler divergence with the uniform distribution over token positions. The derivative of this quantity with respect to an auxiliary temperature gives a notion of the susceptibility of this divergence to changes in the query-key scores.

I feed in two sets of prompts, ones that “activate” the IOI circuit, and similar ones that do not. Tracking the mean statistics of my diagnostics per head over these prompts allows for statistically significant distinguishability between head types (p=0.0002 and p<0.0001 for each diagnostic, respectively).

I also explore cross-head correlations of these diagnostics, which shows promise as a scalable methodology for circuit-discovery, as the analysis only requires a forward-pass. The idea is to eventually cull non-correlated attention heads and test remaining candidates with traditional mechanistic interventions, potentially filling the QK circuit gap currently unaddressed by CLTs.

---

---

## What’s your ideal breakdown of your time in a working week, in terms of hours or % per week spent on research discussions and meetings, coding and code review, reading papers, etc.? *

I consider a productive work week to be 40-60 hours (occasionally more) with the following breakdown:

Coding and code review – 40%
Writing/Presentation preparation – 25%
Reading – 20%
Meetings and discussions – 15%

Writing and coding time can eat into each other, depending on the week. I typically have one (maybe two) “flow states” per day when I get my most labor intensive tasks done. These tend to last 3-6 hours (with small breaks). It is important to me to go on a 10-30 minute walk every day.

---

---

## Additional information

The notebook with my IOI-circuit analysis can be found at https://github.com/peter-fields/peter-fields.github.io/blob/main/notebooks/post2_attention-diagnostics/final/attention_diagnostics_peter.ipynb

My arXiv preprint on temperature tuning in energy-based models can be found at https://arxiv.org/abs/2512.09152.

The blog post linked above is part of an ongoing series. Further posts are planned, developing the cross-head correlation approach introduced in post 2 toward potential scalable circuit discovery.

