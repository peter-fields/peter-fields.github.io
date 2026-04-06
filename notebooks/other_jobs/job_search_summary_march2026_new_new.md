# Job Search Summary — Peter Fields

**Updated: March 24, 2026**

## Profile
PhD-level researcher in statistical physics/biophysics. Primary language Julia; Python/TransformerLens competent. Published arXiv preprint + ICML workshop paper on temperature tuning in protein sequence models. Technical blog (peter-fields.github.io) with two posts on attention diagnostics for mechanistic interpretability. Validated diagnostics on GPT-2-small's IOI circuit with statistically significant results (p=0.0002, p<0.0001). Background bridges protein sector analysis, energy-based models, and transformer interpretability.

**Core value proposition across all roles:** Rigorous ML scientist who can parachute into complex problems — whether studying AI itself or bringing ML to domain-specific challenges — and figure out what's actually going on. The biology work is evidence of this versatility, not a constraint.

---

## Status Updates

- ❌ **Anthropic — Research Scientist, Interpretability:** Rejected after CodeSignal assessment (March 2026). Recruiter noted application and test results were "promising" and encouraged reapplying in ~1 year. Door not closed permanently.
- ❌ **Google DeepMind — all three interp/safety listings** (Interpretability of LLMs, Empowering Humans Using LLMs, AI Safety and Alignment): Confirmed closed as of March 2026. No dedicated mech interp role currently on their board. Set up a job alert at https://job-boards.greenhouse.io/deepmind.

---

## Strategic Note

**Don't limit the search to mechanistic interpretability.** Pure mech interp roles are concentrated at a handful of organizations (Anthropic, OpenAI, DeepMind, a few nonprofits) and the market is thin. Your background supports at least three parallel tracks:

1. **Mechanistic interpretability / AI safety** — the dream roles (OpenAI, ENIGMA, UK AISI, FAR.AI, Anthropic reapply)
2. **Fundamental generative model science** — understanding when and why generative models work, fail, and generalize. Your temperature tuning paper, EBM work, and experimental protein validation are directly relevant here. (Nvidia GenAI4Science, ByteDance Seed, Polymathic AI)
3. **Broad ML research / AI-for-science** — the "parachute scientist" pitch: rigorous quantitative researcher who can drop into complex problems across domains. (BCG X, Argonne, MSR, IBM, Salesforce)

Tracks 2 and 3 also build the experience and publication record that strengthens a future Anthropic reapplication. Don't treat them as consolation prizes — they're genuine fits for your skills.

---

## Action Plan (March 24, 2026)

**Urgent — deadline within 2 weeks:**
- ✅ **LASR Labs Summer 2026** — applied 2026-04-02 (late, past March 30 deadline). 13-week AI safety research program in London.

**This week / next week — highest priority applications:**
- ✅ **OpenAI — Researcher, Interpretability** — applied 2026-03-25
- ✅ **OpenAI — Early Career Cohort** — applied 2026-03-31; starts June 3
- 🔲 **OpenAI — Researcher, Alignment** — strong secondary target, high material reuse
- ✅ **NVIDIA — Fundamental Generative AI (JR2012698)** — applied 2026-03-31
- ✅ **NVIDIA — Fundamental Generative AI (JR2013293)** — applied 2026-03-31
- 🔲 **Stanford ENIGMA** — send CV + one-page interest statement to recruiting@enigmaproject.ai
- 🔲 **FAR.AI — Research Scientist** — no listed deadline, rolling. Mech interp is an explicit research area.
- 🔲 **NYU/Polymathic AI — Postdoctoral Researcher** — HIGH PRIORITY, mech interp of scientific foundation models, rolling, NYC-based

**Next 2–3 weeks — strong fits, apply in parallel:**
- 🔲 **Perplexity — Research Residency** — rolling, $220K/year prorated. Explicitly welcomes physicists. High comp, high profile.
- ❌ **BCG X AISI Postdoctoral Fellow** — CLOSED March 22, 2026. Monitor for future cohorts.
- 🔲 **UK AISI — Interpretability Researcher**
- ✅ **Anthropic Fellows Program (July 2026 cohort)** — submitted 2026-03-30. Response expected in 4-6 weeks.
- 🔲 **Apollo Research — Expression of Interest** — low-effort submission
- 🔲 **Salesforce AI Research** — role posted Feb 2026

**Best-effort / check for openings:**
- 🔲 Microsoft Research New England — check for open roles
- 🔲 IBM Research — check for open roles
- 🔲 Argonne National Laboratory — check for open roles (rolling)
- 🔲 Center for AI Safety (CAIS) — check careers page

**Monitor — no current opening but likely to repost:**
- 👀 Google DeepMind — set job alert, interp roles will cycle back
- 👀 Timaeus — January 2026 deadline passed; watch for next round
- 👀 OpenAI Residency — 2026 apps closed; monitor for future rounds

---

## TIER 1 — Mechanistic Interpretability / Safety (strongest matches)

### OpenAI — Researcher, Interpretability
- **Link:** https://jobs.ashbyhq.com/openai/c44268f1-717b-4da3-9943-2557f7d739f0/application
- **Location:** San Francisco
- **Status:** ✅ Applied 2026-03-25
- **Notes:** "Mechanistic interpretability or spiritually related disciplines." PhD required. Collaborative, curiosity-driven team. Very similar framing to Anthropic role. Develop and publish research on understanding representations of deep networks.
- **Additional Information draft (2026-03-25 — FINAL, submitted):**

> Current methods in interpretability research such as Cross-Layer Transcoders and Sparse Autoencoders require significant compute and training in order to identify features and their attribution graphs. Activation patching and ablation methods require per-head/per-prompt intervention. In recent blog posts I have developed forward-pass-only diagnostics that measure shifts in attention-head distributions across different prompt classes; this has neither of the aforementioned scalability issues. I have done preliminary tests on the IOI circuit: the key innovation is to define two sets of prompt texts (50 circuit-activating and 50 non-activating) over which to track the mean differences in shifts of the attention distribution statistics (KL-divergence with the uniform distribution and a temperature-susceptibility). Circuit heads become discriminable from non-circuit heads in light of these diagnostics (p<0.001).
>
> These methods are inspired by my training in statistical physics: one can extract much structure from distributional properties — a valuable complement to direct intervention. The safety relevance is scalability: if candidate circuit heads can be identified cheaply — no training, no per-head intervention — it becomes more feasible to do per-head circuit-level analysis across many heads and many models on a winnowed set from the initial diagnostics. Whether these observational diagnostics generalize from GPT-2's clean circuits to the messier, more distributed representations in frontier models is an open question, and answering it is exactly the kind of safety-relevant work OpenAI is positioned to pursue.

  **NOTE:** Framing is "scalable circuit head identification," NOT "QK circuit gap" — the diagnostics operate on attention pattern outputs, not the QK mechanism itself.

### OpenAI — Research Scientist / Research Engineer, Early Career Cohort
- **Link:** https://jobs.ashbyhq.com/openai (search "Early Career Cohort")
- **Location:** San Francisco
- **Compensation:** $295K flat
- **Start date:** June 3, 2026 (rolling applications)
- **Status:** ✅ Applied 2026-03-31
- **Notes:** Designed for recent graduates / early-career researchers. Explicitly calls out "mathematician, physicist, or quantitative researcher interested in foundations of intelligence." One-month bootcamp + team matching. Different pitch than Interpretability role — lead with physics-to-AI-research pivot.
- **Additional Information draft (2026-03-21):**

> I recently completed a PhD in physics and am now focused full-time on mechanistic interpretability of neural networks — the kind of pivot this cohort seems designed for. My PhD trained me to extract structure from complex systems using statistical and geometric tools; I'm now applying that instinct directly to understanding how transformers represent and process information.
>
> In the months since graduating, I've developed forward-pass-only attention statistics (KL divergence from uniform, temperature susceptibility) that identify circuit-member heads in GPT-2 without activation patching — a scalability improvement over standard intervention-based methods. These discriminate circuit from non-circuit heads with p=0.0002 and p<0.0001 across prompt types. The statistical physics background behind this approach is documented in my arXiv preprint (arxiv.org/abs/2512.09152); the interpretability work is in an ongoing blog series at peter-fields.github.io.
>
> What draws me to this cohort specifically is the bootcamp and team-matching structure. I'm at an early stage in AI research and genuinely uncertain which directions will be most productive — I'd rather spend a month working across teams and finding where I can contribute most than commit prematurely to a single problem.

### OpenAI — Researcher, Alignment
- **Link:** https://openai.com/careers/researcher-alignment-san-francisco/
- **Location:** San Francisco
- **Status:** 🔲 Apply — strong secondary target
- **Notes:** Direct research role, Safety Systems dept. Mech interp is foundational to alignment; physics PhD maps well. Similar profile to Interpretability role but broader scope. Lead with how understanding circuit structure informs alignment approaches.

### OpenAI — Research Scientist, PhD (General Track)
- **Link:** https://openai.com/careers/research-scientist-phd-san-francisco/
- **Location:** San Francisco
- **Status:** 🔲 Consider — general pipeline, placed on team after hiring
- **Notes:** For recently-completed PhDs with demonstrated research impact via publications. Physics PhD + preprint qualifies. Lower specificity than the Interpretability or Alignment roles — good fallback if those don't progress.

### Stanford ENIGMA — Research Scientist, Interpretability (1-year fixed term)
- **Link:** https://careersearch.stanford.edu/jobs/research-scientist-interpretability-1-year-fixed-term-28732
- **Application:** Also send CV + one-page interest statement to recruiting@enigmaproject.ai
- **Location:** Stanford
- **Salary:** $156K–$180K
- **Status:** 🔲 Apply soon
- **Notes:** Applying mechanistic interpretability to neuroscience models. Circuit discovery, feature visualization, geometric analysis of high-dimensional neural data. Biophysics + interp combo is nearly tailor-made. Requires PhD + 2 years post-PhD research experience.

### UK AI Safety Institute (AISI) — Interpretability Researcher
- **Link:** https://boards.eu.greenhouse.io/aisi/jobs/4387417101
- **Location:** UK
- **Status:** 🔲 Apply soon
- **Notes:** Brand new mech interp team. "How can we tell if a model is scheming?" Access to 5,448 H100s. Mentorship from Geoffrey Irving and Yarin Gal.

### FAR.AI — Research Scientist
- **Link:** https://www.far.ai/careers/research-scientist
- **Location:** Berkeley (remote possible)
- **Status:** 🔲 Apply — rolling, no listed deadline
- **Notes:** Independent AI safety nonprofit. Explicitly lists mechanistic interpretability as a research area (SAEs, interpretable data attribution, probing deception). Welcomes unconventional backgrounds and PhDs in physics. Publishes at NeurIPS, ICLR, ICML. Organized into research pods; encouraged to propose novel directions.

### Meta — AI Research Scientist, Safety Alignment Team
- **Link:** https://www.metacareers.com/profile/job_details/2864638960409054
- **Location:** Menlo Park/NYC
- **Notes:** Within Meta Superintelligence Labs. Broader than mech interp but safety-focused.

### Anthropic — Research Scientist, Interpretability *(reapply in ~1 year)*
- **Link:** https://job-boards.greenhouse.io/anthropic/jobs/4980427008
- **Status:** ❌ Rejected March 2026. Encouraged to reapply after gaining more experience.
- **Notes:** Flagship match. Attention diagnostics address their "Missing Attention Circuits" gap. Keep building interp portfolio; aim to reapply ~early 2027.

### Anthropic — Research Engineer, Interpretability
- **Link:** https://job-boards.greenhouse.io/anthropic/jobs/4020305008
- **Location:** San Francisco
- **Notes:** More engineering-focused variant. Same team. Could be worth a separate application — the RE role emphasizes infrastructure for interp research (activation extraction, steering vectors) rather than the research itself.

### Anthropic — Research Engineer / Scientist, Alignment Science (SF)
- **Link:** https://job-boards.greenhouse.io/anthropic/jobs/4631822008
- **Location:** San Francisco
- **Salary:** $350K–$500K
- **Status:** 🔲 Consider — more engineering-oriented than RS roles
- **Notes:** Empirical AI research with safety focus — robustness testing, multi-agent RL, adversarial training, RSP evaluation. Technical safety, not product. Requires "significant software, ML, or research engineering experience." Python proficiency (interviews in Python). Circuit-level understanding informs safety evaluation work.

### Anthropic — Research Engineer / Scientist, Alignment Science (London)
- **Link:** https://job-boards.greenhouse.io/anthropic/jobs/4610158008
- **Location:** London, UK
- **Status:** 🔲 Consider if open to UK
- **Notes:** Same profile as SF Alignment Science role. ASL-3/ASL-4 safety risks focus.

---

## TIER 1.5 — Interpretability-Adjacent Research (mech interp is part of the work, not the whole job)

### NYU / Polymathic AI — Postdoctoral Researcher, Mechanistic Interpretability of Scientific Foundation Models
- **Link:** Apply via Indeed or NYU CDS careers page (search "Polymathic AI")
- **Location:** New York City (on-site at NYU Center for Data Science)
- **Salary:** ~$120K–$250K range for Research Scientist level; postdoc likely lower
- **Status:** 🔲 Apply — rolling basis
- **Notes:** Focused on generalization, transfer, and mechanistic interpretability of scientific foundation models (astrophysics, fluid dynamics, biology, weather). Interdisciplinary team spanning NYU and Flatiron Institute. Your stat phys + interp background is an excellent match. Based in NYC.

### Apollo Research — Research Scientist (Expression of Interest)
- **Link:** https://www.apolloresearch.ai/careers/
- **Location:** London (primarily)
- **Status:** 🔲 Submit Expression of Interest — 2026 hiring cycle opening
- **Notes:** Combines behavioral model evaluations with applied interpretability. Focus on deceptive alignment and scheming detection. Not pure mech interp but interp-informed. Low-effort EOI submission.

### Center for AI Safety (CAIS) — Research Scientist
- **Link:** https://safe.ai/careers
- **Location:** San Francisco
- **Status:** 🔲 Check current openings
- **Notes:** Leading AI safety research org with large compute cluster. Research cited 16,000+ times. Check for interp-related openings.

---

## TIER 1D — Applied AI Research (not interp, but strong career capital)

### Perplexity — Research Residency
- **Link:** https://jobs.ashbyhq.com/perplexity/adf189e8-5802-4077-aa1f-3668206aacbb
- **Program page:** https://www.perplexity.ai/hub/ai-research-residency
- **Location:** San Francisco or Palo Alto (in-person)
- **Duration:** 3–6 months (confirm current term length when applying)
- **Compensation:** $220,000/year prorated
- **Status:** 🔲 Apply — rolling, accepted until positions filled
- **Notes:** Flagship residency explicitly welcoming researchers from non-traditional AI backgrounds (physicists, cognitive scientists, biochemists). Paired with senior researcher mentors. Access to compute, datasets, and encouragement to publish. Comprehensive benefits + visa sponsorship. Not interpretability-focused — Perplexity's research is search, agentic systems, and human-AI interaction. But excellent comp, high-profile name, and would build the "more experience" Anthropic wants. Resume/CV + cover letter + optional research statement.

### Perplexity — AI Researcher (full-time)
- **Link:** https://jobs.ashbyhq.com/perplexity/8fe61c73-0daf-4432-a47d-44714c1ef764
- **Location:** San Francisco
- **Status:** 🔲 Apply if interested in applied AI research pivot
- **Notes:** Full-time research scientist role. Three teams: Core Research (base model improvement), Deep Research Agent, Comet Agent. Product-facing but real research (publish at conferences, open-source contributions). More of a pivot away from interp than the residency.

---

## TIER 2 — Fellowships & Stepping Stone Programs

### Anthropic Fellows Program (July 2026 cohort)
- **Link:** https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/
- **Duration:** 4 months, paid
- **Status:** 🔲 Apply — rolling applications
- **Notes:** Mechanistic interpretability is a listed focus area. Now elevated in priority: 4 months of focused interp work at Anthropic is exactly the "more experience" the rejection email suggested. Strong path back to a full-time offer.

### LASR Labs — Summer 2026
- **Link:** https://www.lasrlabs.org
- **Location:** London (in-person)
- **Duration:** 13 weeks (July–October 2026)
- **Stipend:** £11,000
- **Status:** 🔲 **APPLY NOW — deadline March 30th**
- **Notes:** Technical AI safety research program. Past projects include interpretability probes, concept extrapolation, automated interpretability. Supervised by researchers from DeepMind, UK AISI, top UK universities. Alumni have gone to UK AISI, Apollo Research, Leap Labs, Open Philanthropy. 4 out of 5 groups in 2023 had papers accepted to NeurIPS workshops or ICLR.

### SPAR — AI Safety Research Fellowship
- **Link:** https://sparai.org/
- **Duration:** 3 months, part-time, remote
- **Notes:** Low commitment, good for building connections in the safety community.

### OpenAI Residency
- **Link:** https://openai.com/residency/
- **Status:** 2026 applications closed. Monitor for future rounds.

---

## TIER 1B — Fundamental AI Research (big labs, studying AI qua AI)

*Lead with interpretability work and general ML research depth. For Nvidia specifically, lead with the temperature tuning / generative model science angle — understanding when and why generative models fail, the interplay between training data, model capacity, and sampling strategy.*

### NVIDIA — Research Scientist, Fundamental Generative AI (New College Grad 2026)
- **Link:** https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Research-Scientist--Fundamental-Generative-AI---New-College-Grad-2026_JR2012698
- **Location:** Santa Clara, CA
- **Salary:** $168K–$305K
- **Status:** 🔲 Apply — STRONG FIT
- **Notes:** Explicitly interested in generative AI for biomolecular design (protein, RNA, small molecule) and GenAI4Science. Your temperature tuning paper is almost tailor-made — rigorous first-principles investigation of generative model behavior on protein EBMs, ground-truth benchmarks, forward/reversed KL tradeoff. Team publishes at top ML venues with real research freedom. Strong mathematical foundation required. **Note:** Team lead Karsten Kreis (karstenkreis.github.io) is a physicist who did his PhD in computational/statistical physics before pivoting to generative AI — very similar trajectory to yours. Consider reaching out directly when applying.

### NVIDIA — AI Safety Scientist, Deep Learning
- **Link:** https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Safety-Scientist--Deep-Learning_JR2012815
- **Location:** Santa Clara, CA
- **Salary:** $120K–$236K
- **Status:** 🔲 Apply
- **Notes:** Applied safety — content safety, fairness, hallucinations, robustness for multilingual/multimodal LLMs. Not interp, but Nvidia works with every major AI company. Unique cross-ecosystem exposure.

### NVIDIA — Senior Research Scientist, Post-Training LLM
- **Link:** Search "Post-Training LLM" at Nvidia careers
- **Location:** Santa Clara, CA
- **Salary:** $160K–$299K
- **Status:** 🔲 Check — may be too senior, but post-training sampling strategy is literally what your paper is about
- **Notes:** Post-training temperature/sampling optimization. Your insight that optimal temperature depends on the interplay between model capacity and data availability transfers directly from protein EBMs to LLMs.

### Microsoft Research — Senior Researcher, ML/AI
- **Link:** https://www.microsoft.com/en-us/research/careers/open-positions/
- **Location:** Redmond WA, Cambridge MA (New England lab), NYC
- **Status:** 🔲 Check open roles — rolling hiring
- **Notes:** Genuinely independent research groups. New England lab has ML & Statistics and Biomedical ML groups. Interpretability and trustworthy AI are live research areas. Look for "Senior Researcher" or "Researcher" titles.

### IBM Research — Research Scientist, AI Foundations
- **Link:** https://www.ibm.com/careers/research
- **Location:** Yorktown Heights NY, Cambridge MA, San Jose CA
- **Status:** 🔲 Check open roles — rolling hiring
- **Notes:** AI Foundations group: novel training algorithms, efficient fine-tuning, architectures. Physics/math backgrounds historically well-regarded. Trustworthy/explainable AI track.

### Salesforce AI Research — Research Scientist
- **Link:** https://careers.salesforce.com/en/jobs/jr214107/research-scientist-salesforce-ai-research/
- **Location:** Palo Alto CA, Seattle WA, San Francisco CA
- **Salary:** $117K–$344K
- **Status:** 🔲 Apply — role posted Feb 2026
- **Notes:** Publishes at top venues. Responsible AI / FATE track maps to interpretability background.

---

## TIER 1C — ML + Broad Domain Science Roles

*Lead with the "parachute scientist" framing.*

### BCG X — AI Science Institute Postdoctoral Fellow
- **Link:** https://careers.bcg.com/global/en/job/57011/AI-Science-Institute-Postdoctoral-Fellow-United-States-BCG-X
- **Location:** Boston, Seattle, or other US cities
- **Salary:** $158,400
- **Duration:** 24 months, with possible transition to full-time
- **Status:** 🔲 Apply — HIGH PRIORITY
- **Notes:** Shape problem formulation and data strategy across bioinformatics, climate, materials, computing. Full BCG infrastructure + support for publishing.

### Argonne National Laboratory — Postdoctoral Appointee / Staff Scientist, AI/ML
- **Link:** https://www.anl.gov/hr/careers
- **Location:** Lemont, IL
- **Status:** 🔲 Check open roles — rolling hiring
- **Notes:** Cross-domain AI (materials, climate, energy, national security). Aurora supercomputer. Named fellowships: Margaret Butler, Walter Massey.

### McKinsey QuantumBlack — Data Scientist / ML Scientist
- **Link:** https://www.mckinsey.com/capabilities/quantumblack/careers-and-community
- **Location:** Multiple US cities
- **Status:** 🔲 Check open roles
- **Notes:** Cross-domain ML. **Caveat:** consulting-flavored — real time on client decks and KPIs.

---

## TIER 3 — Broader AI/Science Roles

### ByteDance Seed — Research Scientist, Generative AI for Science (2026 Start, PhD)
- **Link:** https://seed.bytedance.com/en/direction/ai_for_science
- **Location:** San Jose, CA
- **Status:** 🔲 Apply
- **Notes:** Dedicated AI-for-Science team building biomolecular foundation models for protein structure prediction and de novo protein design. Want backgrounds in diffusion models, geometric deep learning, computational protein design. Your EBM + protein sequence + temperature tuning work is directly relevant. Active hiring for 2026.

### Basis Research Institute (Cambridge, MA)
- **Notes:** Research Scientist, ML/AI. Develop computational theories of intelligence, work with domain experts to solve scientific problems. PhD in physics, math, computational neuroscience, or cognitive science. "Parachute scientist" framing fits well.

- **Google DeepMind, Science Team** — monitor for computational positions

---

## Monitor / Closed / Reapply Later

| Organization | Role | Status | Next Step |
|---|---|---|---|
| Anthropic | Research Scientist, Interp | ❌ Rejected March 2026 | Reapply ~early 2027 with stronger portfolio |
| DeepMind | Interp of LLMs | ❌ Closed | Job alert set; reapply when reposted |
| DeepMind | Empowering Humans Using LLMs | ❌ Closed | Job alert set |
| DeepMind | AI Safety and Alignment | ❌ Closed | Job alert set |
| Timaeus | Research Scientist/Engineer | ❌ Jan 2026 deadline passed | Watch for next round |
| OpenAI Residency | Residency | ❌ 2026 apps closed | Monitor for future rounds |
| MATS Summer 2026 | Fellowship | ❌ Jan 18 deadline passed | — |

---

## Honorable Mentions (peruse when time allows)

- **FutureHouse** (futurehouse.org/fellowship) — AI-for-science nonprofit in SF. Monitor for 2027.
- **Schmidt Futures — AI in Science Postdoctoral Fellowship** — university-hosted, industry-funded, broad domain scope.
- **Allen Institute for AI (AI2)** — Seattle-based nonprofit. Serious ML research culture.
- **Sandia / Lawrence Berkeley / Oak Ridge National Labs** — similar to Argonne, different locations.
- **Flagship Pioneering / Recursion** — ML-for-biology; only if appetite for bio-specific work returns.
