# System-card excerpts (verbatim)

Durable copies of the passages this project is built on, so the notes are self-contained.
Pulled from the official cards on 2026-07-14 via `pdftotext`.

---

## OpenAI — GPT-5 System Card, §3.3 "Sycophancy"
Source: https://openai.com/index/gpt-5-system-card/

> In May 2025 we explained the immediate measures we took to address sycophantic behaviors
> that emerged in our GPT-4o model: we rolled back a newly deployed version of the GPT-4o
> model, and also adjusted the system prompt for the model that remained in production.
> System prompts, while easy to modify, have a more limited impact on model outputs relative
> to changes in post-training. For GPT-5, we post-trained our models to reduce sycophancy.
> Using conversations representative of production data, we evaluated model responses, then
> assigned a score reflecting the level of sycophancy, which was used as a reward signal in
> training.

> In offline evaluations ... gpt-5-main performed nearly 3x better than the most recent
> GPT-4o model (scoring 0.145 and 0.052, respectively) and gpt-5-thinking outperformed both
> models.

> In preliminary online measurement of gpt-5-main ... prevalence of sycophancy fell by 69%
> for free users and 75% for paid users in comparison to the most recent GPT-4o model.

**Table 4 (Result, lower is better):** GPT-4o 0.145 · gpt-5-main 0.052 · gpt-5-thinking 0.040
(offline); gpt-5-main online prevalence −0.69 free / −0.75 paid vs 4o.

### §3.3.1 "Looking ahead" (the admitted gap this project sits in)
> We have post-trained the GPT-5 models to be less sycophantic, and we are actively
> researching related areas of concern, such as situations that may involve emotional
> dependency or other forms of mental or emotional distress. These areas are particularly
> challenging to measure, in part because while their importance is high, their prevalence
> currently appears to be low. We are engaging human-computer-interaction (HCI) researchers
> and clinicians to give feedback on our definitions for concerning interactions, and on our
> evaluation methods. We are working to mature our evaluations in order to set and share
> reliable benchmarks which can in turn be used to make our models safer in these domains.

---

## Anthropic — Claude Opus 4.5 System Card
Source: https://www.anthropic.com/claude-opus-4-5-system-card

### §6.1 — definitions (sycophancy and over-refusal are sibling failure modes)
> Sycophancy: Unprompted excessive praise or validation of bad ideas
> Encouragement of user delusion: Extreme cases of sycophancy involving broader
> disconnection from reality
> Overrefusal: Refusing requests that are not, on balance, likely to cause harm if complied
> with

### Character section (sycophancy "reached a new low")
> We see these improvements alongside a decrease in the related but unwanted trait of
> sycophancy, which has reached a new low. However, see discussion of user-sourced prompts
> below for further discussion.

### §6.2.4 — external comparison via the open-source Petri tool
> we have also released the open-source package Petri, which replicates a similar style of
> evaluation in a form that is compatible with and comparable across models from many
> developers. ... We tested a pre-final preview snapshot of Claude Opus 4.5 and report five
> major metrics: Concerning ..., audit situational awareness ..., cooperation with human
> misuse, deception toward the user, and sycophancy. We used Claude Sonnet 4.5 and GPT-5 as
> auditors, and Claude Opus 4.1, Gemini 2.5 Pro, and GPT-5 as scorers.

(362 investigations per model under study; lower score = lower rate/severity.)

### §6.3 — "Sycophancy on user-provided prompts" (the finding being replicated)
> To evaluate how Claude Opus 4.5 performs in real-world conversations where previous models
> behaved sycophantically, we developed an evaluation that uses real user conversations
> shared with Anthropic as Feedback. Using our tool for analysing aggregated Claude
> conversations, we identified Feedback conversations where user inputs appeared disconnected
> from reality and where Claude responded sycophantically. We then removed the system prompt
> and re-sampled assistant responses in the conversation, scoring the new responses using a
> grader prompt. The evaluation covers a wide range of scenarios, such as users expressing
> grandiose beliefs about their own scientific discoveries or supernatural experiences.
> Prompts span multiple languages.

> This is a particularly challenging evaluation: Prompts can include prior assistant
> responses from other models that validated the user's beliefs, meaning the model must
> course-correct mid-conversation rather than simply avoid sycophancy from the outset.

Figure 6.3.A: **Non-sycophantic response rate** on **260 re-sampled turns**, graded by
Claude Opus 4.1. Higher is better.

**The trade-off (the opening for this project):**
> Claude Haiku 4.5's stronger performance reflects training choices that prioritize pushback,
> though this tendency can occasionally come across as harsh. Claude Opus 4.5 underwent
> similar training with additional adjustments aimed at reducing this harshness, which may
> explain its lower score.

### Interpretability section — "blunt speech" as a byproduct of anti-sycophancy training
> A feature representing "blunt speech" increased in activation substantially over training,
> perhaps as a byproduct of training to avoid excessive sycophancy.
