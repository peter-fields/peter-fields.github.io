---
name: lasr-app
description: LASR Labs Summer 2026 application — submitted answers, assessment status
type: project
---

# LASR Labs Summer 2026 Application

**Status:** Assessments completed. Decisions expected early May. Interview invitations expected early May.
**CodeSignal (ML coding):** Taken 2026-04-22. Covered kNN, k-Means, Decision Tree, GMM, Matrix Normalization, Bagging, Forward Prop + Module 2 string/array. Hit numpy shape bug on gradient descent, didn't finish last problem.
**Airtable (AI safety research critique):** Taken 2026-04-23. Paper was about LLM introspection. Felt "better-ish" than coding. Debrief pending.

---

## Why are you a good fit for LASR Labs? (100 words max)

My background in statistical physics, biophysics, and information theory makes me well-positioned for frontier AI safety research; LASR Labs would actualize my potential. Recently, I ported concepts from theoretical neuroscience (doi/10.1073/pnas.2313676121) to mechanistic interpretability. I tracked statistics of attention heads over different kinds of prompt classes (like tracking neural population statistics over different visual stimuli) in order to see which heads change their behavior under varying conditions (like revealing stimulus-independent structure in the retina). I found statistically significant signatures that separated the circuit from non-circuit heads for GPT-2 small's IOI circuit (p<0.001).

---

## Technical artifact

**Link:** https://github.com/peter-fields/temp-tune

This repo is associated with my recent arxiv preprint that explores the connection between generative model sampling techniques, model bias due to limited data and choice of objective function, and properties of the ground truth distribution. It does so through a physicists' lens, navigating theoretical subtlety through simple toy examples–with all simulations and analysis coded end-to-end in Julia.

---

## AI safety idea you disagree with

I disagree with the notion that sycophancy is merely an implementation problem. Clearer rewards and instructions, better evaluations for RLHF---these have shown success in improving other failure modes. But the problem may be deeper; it persists across models and labs. When a model is trained to help so long as a user's intent is deemed benign, flattery is a natural avenue to ensure apparent helpfulness. Pushing back requires a positive good worth pushing back for. Negative objectives may not be enough. Sycophancy may then be seen as an inductive bias of training objective for alignment. This possibility deserves more attention.
