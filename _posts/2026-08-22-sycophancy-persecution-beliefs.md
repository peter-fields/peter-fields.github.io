---
title: "Unsubstantiated beliefs of persecution can elicit sycophancy from LLMs"
layout: single
author_profile: false
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
tags: [ai-safety, sycophancy, evals, llm, calibration]
excerpt: "Frontier models resist crank science and supernatural claims, but readily validate persecution beliefs built on thin evidence. A matched-pair design measures sycophancy as miscalibration."
---


## The TL;DR

{% include figure image_path="/assets/images/posts/syconot/tldr_haiku_vs_opus.png" alt="Validation rates for Haiku 4.5 and Opus 4.5 across crank/supernatural, warranted persecution, and unwarranted persecution conversations." %}

When I gave models multi-turn conversations---wherein users supplied delusional beliefs and previous models had already responded sycophantically---I found that newer models were capable of turning the conversation around and pushing back against delusions like crank scientific theories or supernatural happenings. However, when models had to push back based on evidence *supplied by the user*, Haiku 4.5 was much better at pushing back when evidence was thin, whereas Sonnet and Opus 4.5 struggled. This tracks a similar result from Anthropic's Opus 4.5 system card, where Haiku 4.5 had better non-sycophantic rates in a similar evaluation.

This project was done as part of the BlueDot Technical AI Safety Project Sprint.


## Anthropic showed Haiku 4.5 most readily resists affirming delusional beliefs

In the [system card](https://www-cdn.anthropic.com/bf10f64990cfda0ba858290be7b8cc6317685f47/Claude%20Opus%204.5%20System%20Card.pdf) for Opus 4.5, section 6.3, Anthropic ran an experiment where they gave it (and other models) real conversation histories from previous user/model interactions in which the users expressed beliefs that appeared disconnected from reality. The previous models, which were older than the 4.5 generation being tested, had responded sycophantically for several turns of the conversation. 

When these older, sycophantic conversations were loaded into the newer models (stripped of their system prompts), Anthropic tested to see if said newer models could successfully redirect and push back against the previously sycophantic affirmations of the user beliefs.  

The delusional beliefs tested covered a "wide range of scenarios,
such as users expressing grandiose beliefs about their own scientific discoveries or
supernatural experiences." These conversations were given to Opus/Sonnet/Haiku 4.5 and Opus 4.1, with Opus 4.1 used as a grader with a given rubric. The result is shown below.

{% include figure image_path="/assets/images/posts/syconot/system_card_result.png" alt="Haiku 4.5 resists sycophancy most effectively. Result from Opus 4.5 system card." %}

As Anthropic notes, "Claude Haiku 4.5’s stronger performance reflects training choices that prioritize pushback, though this tendency can occasionally come across as harsh. Claude Opus 4.5 underwent similar training with additional adjustments aimed at reducing this harshness, which may explain its lower score."

## Unsubstantiated claims of persecution by users track the above result

Reproducing the above result is difficult; Anthropic did not release the real user conversations for privacy reasons (totally understandable, of course). However, when I created (with the help of Claude Opus 4.8 and 5) 9-turn conversations with delusional beliefs about crank science or supernatural happenings and already-sycophantic responses from the model, and gave these conversations to Haiku/Opus/Sonnet 4.5 (Opus 4.1 is no longer available through the API), each model performed well in pushing back against such beliefs. 

The conversations actually used by Anthropic may have been longer, and the delusional beliefs more subtle than the ones I was able to create. For example, one conversation had a (fictitious) user discuss their grandiose beliefs about being able to unify the theories of gravity and electromagnetism. Haiku/Opus/Sonnet 4.5 were all able to push back easily. 

{% include figure image_path="/assets/images/posts/syconot/crank_control.png" alt="All three models validated zero of the crank science and supernatural conversations." %}

However, one category of conversation history was able to elicit sycophantic responses, and furthermore, qualitatively reproduce the asymmetry in performance as shown by Anthropic, where Haiku resists sycophantic affirmation best. 

The key was to give conversations where the user supplied all information about the situation. "My manager fired me, and he was always out to get me" or "My things keep going missing, and I'm sure it's my roommate." Naturally, one would expect an LLM to push back adequately on beliefs where it already has ground to stand on. Unifying theories in physics is quite difficult and rare, and there is certainly much in an LLM's training data about the infeasibility of unifying gravity and electromagnetism. 

Care must be taken. When a model cannot verify information from a user independently, it might actually be the case that the user is being persecuted. To control for this, I developed matched pairs for each scenario. For example, for the recently-fired employee who was convinced that their manager was out to get them, in the control prompt this belief is largely based on thin evidence that the manager actually felt threatened by the employee's competence, whereas in the test prompt, the employee reports information from third parties that corroborate the manager's insecurity about status and the cover story of restructuring to motivate the firing. 

For these matched-pair scenarios, non-sycophancy then becomes a matter of validating the user's belief *only when the evidence warrants it.*

The figure below gives the full result.

{% include figure image_path="/assets/images/posts/syconot/calibration_bars.png" alt="Validation rate by model, warranted (solid) vs unwarranted (hatched) persecution beliefs." %}

Ideally we want validation to be high when the evidence warrants it (high solid bar) and low when it does not (small transparent striped bar). We can see that Haiku is the only model whose responses adequately tracked the evidence. This is in line with Anthropic's aforementioned result: Haiku 4.5 does better where Sonnet and Opus 4.5 struggle to resist sycophantic responses.
 
## Methodology

The Opus 4.5 system card defines sycophancy as "unprompted excessive praise or validation of bad ideas."

Sharma et al. (2023), in [*Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548), characterize sycophancy as a model tailoring its responses to match a user's stated beliefs over truthful ones. That framing is consonant with our usage here: the failure is not agreement as such, but agreement that is not backed by the evidence.

Using this definition of sycophancy informs our experiment. It is only sycophantic to validate an idea if the evidence does not warrant it; our matched-pair set-up does just this. It is also important that the validation be unprompted. If a person truly believes they are being persecuted and is deluded about this belief, then one would expect that they would not ask for validation. Each conversation therefore ends with a rant by the user. The user makes no request for validation from the model, and expresses no desire for correction.

Additionally, across all sets of prompts (crank delusional beliefs, warranted persecution beliefs, and unwarranted delusional beliefs), the conversations were 9 turns long (5 from the user and 4 from the model, with the last entry in the history from the user).

These histories were then given to each model under experimentation. The responses were graded by Opus 4.8, with a rubric that instructed the judge to check only whether the user's belief was validated. Pushback, changing the topic, or simply not engaging with the belief were not counted as sycophancy. 

A separate script was run to check if the conversation warranted the persecution belief. It was important to separate these judgements. If the judge was scoring both warrant and validation and had access to the test model's reply, it could inadvertently mark a belief as "warranted" when the test model validated, and unwarranted when the test model refrained from validation. Separating out these judgements protects against this circularity. 

One may argue that we are cherry-picking our scenarios to elicit the sycophantic behaviors. We note, however, that in Anthropic's own evaluation, the user/model conversations were selected *because* they caused model failure. Additionally, Haiku was the model that achieved the best performance; the smaller model, with a different post-training protocol, succeeded where bigger models failed. The fact that the evals developed here show similar results across multiple models suggests that the prompt set used is not idiosyncratic to one model, but actually uncovers a real generalizable phenomenon---namely the propensity of models to validate beliefs when evidence does not warrant it, and furthermore, that this propensity does not necessarily correlate with model size but more so with post-training. 

Each of the three prompt types contains 12 scenarios, and every scenario was run 3 times per model (sampling at the default temperature), giving 36 responses per model per prompt type — 108 per model in total, and 324 graded responses across the three models. 

You can find all prompts, the grading rubrics, and the raw grades [here](https://github.com/peter-fields/persecution-belief-sycophancy-in-llms).

## A larger question

When working on this, I began to think, "Should we be allowing models to weigh in on situations in users' lives of which the model has no ability to verify that information, especially for sensitive situations?"

Ultimately, for this project I decided to assume that a given user was giving a trustworthy account of the evidence. Sycophancy was then judged based on whether this evidence was good. The question of whether the evidence was *reliable* was simply too tricky to incorporate, at least on a first pass.

But when a person is truly delusional about the state of a situation, it seems very likely they are willing to make up evidence, or at the very least grossly misrepresent what they do and do not know. 

To make an analogy, imagine a company that employed several "20 minute therapists." This fleet of therapists is authorized to give advice to clients based on one limited interaction, often only via brief messaging. These therapists need not have any professional certification, have no prior relationships with clients, no medical records or reports on clients' health, and no intent to build rapport over an extended period of sessions thereafter. The company that employs this fleet of therapists has no liability for any resulting harm caused to clients by interactions with any of this fleet of "pocket therapists."

I think you see where I am going with this... In any case, just food for thought for the reader who was patient and thoughtful enough to make it to the end of my blog. 😊
