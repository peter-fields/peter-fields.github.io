---
name: astra-app
description: Constellation Astra Fellowship application — submitted answers, May 2026
type: project
originSessionId: f1b97da7-2faf-4258-8092-c796d88b1498
---
# Constellation Astra Fellowship Application

**Program:** September 14, 2026 – February 5, 2027. Full-time (40 hrs/week). Berkeley (Constellation) or London (LISA).
**Status:** Submitted 2026-05-04 (13 min after midnight; within "Anywhere on Earth" window — valid until noon UTC May 4).
**Workstream:** Empirical Research
**Offers expected:** July 25, 2026

---

## Accomplishments

My arXiv preprint (arxiv.org/abs/2512.09152) explores the connection between generative model sampling techniques, model bias due to limited data and choice of objective function, and properties of the ground truth distribution — all coded end-to-end in Julia (https://github.com/peter-fields/temp-tune). This project formed the core of my dissertation research. The research question was motivated by perplexing results within generative modeling for protein design. I designed and built two simplified toy systems that reproduced all the pertinent phenomenology of the original biological problem. The answer to this puzzle for this specific system bore implications for systems beyond biology.

---

## Why Astra?

I think building a thing we do not fully understand seems rife with unknown unknowns—unquantifiable risk. For this reason, I am interested in pursuing a career in mechanistic interpretability research.

I am interested in probing models for knowledge that may be used by bad actors (terrorists building weapons, hackers building malware for example) and putting in anti-jailbreaking interventions to prevent access to those capabilities.

More broadly, I would like to generate insights from interpretability research that can be leveraged for better behavioral interventions to ensure alignment.

Having recently completed my PhD in physics, which focused on applications of statistical physics to understanding machine learning and biology, my theoretical and analytical toolkit is well-suited for observational and data-driven approaches to understanding emergent behaviors in systems such as AI.

The Astra Fellowship would afford me availability to a research ecosystem that is central in the safety research community, and would help situate my expertise within that environment.

---

## Empirical Research

My background in statistical physics and information theory gives me an analytical toolkit and mindset for understanding emergence and information processing in complex systems. I have already begun to port ideas from my area of research to mechanistic interpretability. In a recent blog post, using ideas from info theory and statistical physics, I motivated metrics for measuring attention head responses to different text-prompts, and found statistically significant signatures that separated IOI circuit heads from non-circuit heads in GPT-2 small.

There are three broad directions of research I'd be interested in pursuing in interpretability:

**(1)** A continuation of the prompt-response analysis outlined above. The analysis only requires forward-passes (no patching, ablations, or SAE/CLT training) and shows promise for identifying correlated computing structures within LLMs at scale.

**(2)** Developing a normative framework for understanding computation in superposition: how it works and what are the necessary ingredients (architecture/data) for its development. I have a number of ideas of where to begin for this direction, but all of them start by building on a recent paper from Bauer and Bialek (https://arxiv.org/html/2512.23531v1) that shows many capacity-limited channels optimally encode information about a signal by being individually ambiguous, but informative in the aggregate.

**(3)** Behavioral interventions and interpretability research that involve direction analysis of the activations in the residual stream usually consider distances among these vectors in Euclidean space. However, the model itself compares residual stream vectors via the bilinear forms that are the QK tensors. Though these are not metrics themselves, they still determine residual stream vector comparison. It is an interesting empirical question whether or not meaningful metric (or perhaps metrics per layers) could be extracted from the symmetric parts of the QK tensors across attention heads.

With regards to this last research direction, I would especially be interested in applying these ideas to work done by Owain Evans on persona vectors and activation oracles. I think the former could benefit from a nuanced understanding of the geometry of activation space, especially if certain activations project onto undesirable persona-directions under non-Euclidean metrics.

---

## Key Framing Notes (for future reference)

- Empirical stream is heavy on AI control/scalable oversight — framed mech interp as complementary tool
- Owain Evans is primary mentor pitch (persona vectors, activation oracles, residual stream geometry)
- Three directions pitched roughly equally: prompt-contrast, Bauer-Bialek superposition theory, W_QK implicit metric
- Metric idea connected concretely to Evans's persona vectors work
- Did NOT pitch Ethan Perez — confirmed Astra mentor (Henry Sleight comment on his AF post), but explicitly excludes interpretability: "highly experimental LLM alignment research, excluding interpretability." His scope: scalable oversight, adversarial robustness, CoT faithfulness, model organisms of misalignment.
- Wrote without AI assistance on the substantive answers per Astra's recommendation
- Constellation has access to Anthropic Fellows application — avoided verbatim repetition
