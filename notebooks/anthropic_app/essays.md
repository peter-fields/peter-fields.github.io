---
title: anthropic app essay questions
---

Why Anthropic?

I strongly agree with Dario Amodei’s assessment that the time to understand AI is before its advancement to a level beyond our understanding, that is: right now. How LLMs may generalize to contexts not explicitly trained on, the fundamental causes of deception and hallucination, jailbreaking methods yet unknown… all such questions and concerns are paramount. 

From Anthropic’s track record (research-first culture, commitment to transparency, explicit safety and ethical concerns) I trust it to value safety over unfettered progress, especially in sensitive and high risk contexts, if our answers to the aforementioned questions are less than satisfactory. Technological advancement has often left society to catch up with its repercussions, but there is no reason this should be our default.

My experience in statistical physics, with its applications in machine learning and biology, makes me strongly poised to approach such questions with technical prowess and curiosity. I have a firm intuition for the fundamental workings of many-component systems, from perspectives in information geometry to practical applications in real systems. Working with such real systems (proteins and neuroscience) has given me a sense for when observational analysis is merited and how this informs direct experimental intervention.  Furthermore, it has developed my intuition for the tensions between persistent idiosyncrasies of a system and its universal and general characteristics. 

Such analyses have inspired me to pursue directions in interpretability research.  I have already begun: developing diagnostic tools for attention head-circuitry and validating on the known IOI circuit in GPT-2. Working at Anthropic would afford me the opportunity to scale up this research to frontier models, and to do so in a rigorous and responsible environment. 

Why do you want to work on the Anthropic interpretability team? (We value this response highly - great answers are often 200-400 words. We like to see deeper and more specific engagement with Anthropic and our interpretability agenda than simply interest in AI or LLMs.)*

Anthropic’s papers “Circuit Tracing” and “On the Biology of a Large Language Model” develop techniques that intervene at the level of the MLPs between attention layers. These cross-layer transcoders help build attribution graphs that allow for tracing feature-to-feature computation through MLPs: the OV circuits. The attention mechanisms themselves are “frozen in,” however, leaving more questions open. Why do certain token positions interact with others within the attention-heads and how does this contribute to the residual stream? How do the QK circuits give rise to the attention patterns? Query-key scores can be understood in terms of the feature basis discovered by CLTs. But this analysis can only operate on a per-prompt position pair basis: a computationally taxing process.


I have developed diagnostic tools for distinguishing circuit heads from non-circuit heads. Rather than consider heads on a per-prompt basis, I track statistics of heads over several prompts of varying types. These quantities are the KL-divergence of a head’s softmax distribution with the uniform distribution and the derivative of this quantity with respect to an auxiliary temperature parameter, giving a notion of attention selectivity and stability, respectively, and requiring only a forward-pass to calculate. In the IOI circuit on GPT-2, I have shown statistically significant shifts in circuit heads when switching between circuit-activating IOI prompts and similar yet non-activating non-IOI prompts.

This work was inspired by my work in statistical physics/machine learning for applications in biology, where observational methods help avoid costly interventions while still revealing pertinent underlying structure. These analyses are not only analogous to my recent work in interpretability; covariance of these diagnostics between heads (and measured across prompts) has shown potential for a scalable method of culling heads down to fewer circuit candidates—as it only requires a forward-pass and no further intervention. 

More broadly, I am interested in studying how to scale-up circuit discovery methods as well as the hypothesis that LLMs often perform reasoning via reservoir computing. If more complicated tasks employ more distributed computation (and therefore less circuit-like), then my toolkit for understanding such collective dynamics is well-suited for this research direction.

The interpretability architecture group interests me as well. Are there ways to promote circuit development while training? One concrete direction: explicitly constraining how selective attention heads are allowed to be—with a progressively weakening constraint with layer depth. This could promote circuit specialization by design, rather than purely relying on emergence via training. The idea follows directly from work I've been developing on the information-theoretic structure of attention.

Please briefly describe the technical achievement you’re most proud of, including what you did, how you collaborated with others (if relevant), and its significance or impact.

My recent arXiv preprint (under review at Physical Review Research), “Understanding temperature tuning in energy-based models” is my proudest technical achievement. Energy-based models are used to generate novel protein sequences, ones that function in vivo. However, they typically employ a post-training temperature tuning of the learned energy function in order to get these functional sequences. My work brought this heuristic into the light of a more rigorous, principled understanding.

I did experiments on simple ground truth models and linked how finite sampling, standard objective functions, and the structure of real-world-data state spaces (functional states greatly outnumbered by nonfunctional states) create the bias that temperature tuning corrects. The analysis also indicated when it would be advantageous to raise instead of lower temperature. 

My advisor, Stephanie Palmer, and my main collaborators David Schwab and Wave Ngampruetikorn, saw the significance of my analysis before I did and pushed me to synthesize it into a coherent narrative, to understand its implications within the broader research community, and to take pride in this work that I initially undersold (to myself most of all).  

To the best of my knowledge, there was not a theoretical framework within which to understand the separate yet related phenomena of finite-sample/objective function inductive biases, metrics of generative performance, ground truth data-generating processes, and generative modeling sampling tricks. My work laid theoretical groundwork to this end. 

Please share a link to the piece of work you've done that is most relevant to the interpretability team, along with a brief description of the work and its relevance.*

https://peter-fields.github.io/attention-diagnostics/

This blogpost utilizes two diagnostics I developed to analyze the behavior of the IOI-circuit in GPT-2. The selectivity of the softmax distribution of a given head for a given prompt is measured via its Kullback-Leibler divergence with the uniform distribution over query-key scores. The derivative of this quantity with respect to an auxiliary temperature gives a notion of the stability of this divergence to changes in the query-key scores.

I feed in two sets of prompts, ones that “activate” the IOI circuit, and similar ones that do not. Tracking the mean statistics of my diagnostics per head over these prompts allows for statistically significant distinguishability between head types (p=0.0002 and p<0.0001 for each diagnostic, respectively). 

I also begin work that tracks correlations of these diagnostics between heads, which shows promise as a scalable methodology for circuit-discovery, as the analysis only requires a forward-pass. The idea being to eventually cull non-correlated attention heads and test remaining candidates with traditional mechanistic interventions, potentially filling the missing attention gap currently unaddressed by CLTs.

What’s your ideal breakdown of your time in a working week, in terms of hours or % per week spent on research discussions and meetings, coding and code review, reading papers, etc.? *

I consider a productive work week to be 40-60 hours (occasionally more) with the following breakdown:

Coding and code review – 40%
Writing/Presentation preparation – 25%
Reading – 20%
Meetings and discussions – 15%

Writing and coding time can eat into each other, depending on the week. I typically have one (maybe two) “flow states” per day when I get my most labor intensive tasks done. These tend to last 3-6 hours (with small breaks). It is important to me to go on a 10-30 minute walk every day.

Additional information

The notebook with my IOI-circuit analysis can be found at https://github.com/peter-fields/peter-fields.github.io/blob/main/notebooks/post2_attention-diagnostics/final/attention_diagnostics_peter.ipynb

I mainly code in Julia, but have a working proficiency in Python, and I am currently improving my knowledge of TransformerLens. 

I am deeply interested in the philosophy of language: one more reason I find language models fascinating.
