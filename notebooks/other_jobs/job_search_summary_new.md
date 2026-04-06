# Job Search Summary — Peter Fields

## Profile
PhD-level researcher in statistical physics/biophysics. Primary language Julia; Python/TransformerLens competent. Published arXiv preprint + ICML workshop paper on temperature tuning in protein sequence models. Technical blog (peter-fields.github.io) with two posts on attention diagnostics for mechanistic interpretability. Validated diagnostics on GPT-2-small's IOI circuit with statistically significant results (p=0.0002, p<0.0001). Background bridges protein sector analysis, energy-based models, and transformer interpretability.

**Core value proposition across all roles:** Rigorous ML scientist who can parachute into complex problems — whether studying AI itself or bringing ML to domain-specific challenges — and figure out what's actually going on. The biology work is evidence of this versatility, not a constraint.

---

## Action Plan (as of March 2026)

**Right now — top priority:**
- ✅ Anthropic application submitted
- 🔲 **Complete Anthropic coding assessment** — this is the most concrete lead; everything else is secondary

**Next 2 weeks — best effort, in rough priority order:**
- 🔲 BCG X AISI Postdoctoral Fellow — apply (HIGH PRIORITY after Anthropic)
- 🔲 OpenAI interpretability — high material reuse from Anthropic app, should be quick
- 🔲 Salesforce AI Research — role currently open (posted Feb 2026), apply soon
- 🔲 DeepMind interpretability of LLMs
- 🔲 Stanford ENIGMA — send CV + one-page interest statement to recruiting@enigmaproject.ai
- 🔲 UK AISI
- 🔲 Microsoft Research New England — check for open roles
- 🔲 IBM Research — check for open roles
- 🔲 Argonne — check for open roles (rolling, Chicago)

**Note:** There's only so much time. The Anthropic assessment takes priority. If some of these apps close before getting to them — that's okay. Do your best.

---

## TIER 1 — Interpretability/Safety (strongest matches)

### Anthropic — Research Scientist, Interpretability
- **Link:** https://job-boards.greenhouse.io/anthropic/jobs/4980427008
- **Location:** San Francisco (remote possible for exceptional candidates)
- **Status:** ✅ Application submitted — coding assessment in progress
- **Notes:** Flagship match. Attention diagnostics address their stated "Missing Attention Circuits" gap in CLT attribution graphs.

### Anthropic — Research Engineer, Interpretability
- **Link:** https://job-boards.greenhouse.io/anthropic/jobs/4020305008
- **Location:** San Francisco
- **Notes:** More engineering-focused variant. Same team.

### OpenAI — Researcher, Interpretability
- **Link:** https://openai.com/careers/researcher-interpretability-san-francisco/
- **Location:** San Francisco
- **Status:** 🔲 Apply next — high reuse of Anthropic materials
- **Notes:** "Mechanistic interpretability or spiritually related disciplines." PhD required. Very similar to Anthropic role.
- **Additional Information draft (2026-03-21):**

> Current mechanistic interpretability relies on activation patching to identify circuit components — but patching is expensive and doesn't scale to the circuits that matter in frontier models. I've developed forward-pass-only attention statistics (KL divergence from uniform, temperature susceptibility) that read out circuit structure directly from attention representations, without interventions. Applied to the IOI circuit in GPT-2, these statistics discriminate circuit from non-circuit heads with p=0.0002 and p<0.0001 across prompt types.
>
> The statistical physics background — using ensemble-level observational statistics to reveal structure that per-instance measurements miss — is documented in my arXiv preprint (arxiv.org/abs/2512.09152) and in an ongoing blog series at peter-fields.github.io.
>
> The practical payoff is scalability: if attention representations encode circuit membership in a way that's legible without interventions, it becomes feasible to monitor how head specialization evolves as models scale, or to audit deployed models without running counterfactual experiments. Testing whether these statistics generalize from GPT-2's clean circuits to the messier, more distributed representations in frontier models is exactly the kind of work OpenAI is uniquely positioned to pursue.

  **NOTE**: framing is "scalable circuit head identification," NOT "QK circuit gap" — the diagnostics operate on attention pattern outputs, not the QK mechanism itself. Don't claim to address the QK gap with Posts 1-2.

### Google DeepMind — Research Scientist, Interpretability of LLMs
- **Link:** https://startup.jobs/research-scientist-interpretability-of-llms-deepmind-6412235
- **Location:** Seattle
- **Status:** 🔲 Apply soon
- **Notes:** "Practical interpretability" — controllability as evaluation criterion. Diagnostics frame well as observational/practical tools.

### Google DeepMind — Research Scientist, Empowering Humans Using LLMs
- **Link:** https://job-boards.greenhouse.io/deepmind/jobs/6688786
- **Location:** Seattle
- **Notes:** Broader than pure mech interp. Team includes PhDs in interpretability, cognitive science, Bayesian theory.

### Google DeepMind — Research Scientist, AI Safety and Alignment
- **Link:** https://startup.jobs/research-scientist-ai-safety-and-alignment-deepmind-4967707
- **Location:** Multiple (including NYC)
- **Notes:** Interpretability is a core component alongside robustness, evaluations, monitoring.

### Stanford ENIGMA — Research Scientist, Interpretability (1-year fixed term)
- **Link:** https://careersearch.stanford.edu/jobs/research-scientist-interpretability-1-year-fixed-term-28732
- **Application:** Also send CV + one-page interest statement to recruiting@enigmaproject.ai
- **Location:** Stanford
- **Salary:** $156K–$180K
- **Status:** 🔲 Apply soon
- **Notes:** Applying mechanistic interpretability to neuroscience models. Biophysics + interp combo is nearly tailor-made here.

### UK AI Safety Institute (AISI) — Interpretability Researcher
- **Link:** https://boards.eu.greenhouse.io/aisi/jobs/4387417101
- **Location:** UK
- **Status:** 🔲 Apply soon
- **Notes:** Brand new mech interp team. "How can we tell if a model is scheming?" Access to 5,448 H100s. Mentorship from Geoffrey Irving and Yarin Gal.

### Meta — AI Research Scientist, Safety Alignment Team
- **Link:** https://www.metacareers.com/profile/job_details/2864638960409054
- **Location:** Menlo Park/NYC
- **Notes:** Within Meta Superintelligence Labs. Broader than mech interp but safety-focused.

---

## TIER 1B — Fundamental AI Research (big labs, studying AI qua AI)

*Lead with interpretability work and general ML research depth. These orgs want people who can push the state of the art on AI itself.*

### Microsoft Research — Senior Researcher, ML/AI
- **Link:** https://www.microsoft.com/en-us/research/careers/open-positions/
- **Location:** Redmond WA, Cambridge MA (New England lab), NYC
- **Status:** 🔲 Check open roles — rolling hiring
- **Notes:** Genuinely independent research groups that publish at top venues. New England lab (Cambridge, MA) has ML & Statistics and Biomedical ML groups. Interpretability and trustworthy AI are live research areas. Look for "Senior Researcher" or "Researcher" titles — not "Applied Scientist" which is more product-facing. Strong research autonomy for an industry lab.

### IBM Research — Research Scientist, AI Foundations
- **Link:** https://www.ibm.com/careers/research
- **Location:** Yorktown Heights NY, Cambridge MA, San Jose CA
- **Status:** 🔲 Check open roles — rolling hiring
- **Notes:** AI Foundations group works on fundamental ML: novel training algorithms, efficient fine-tuning, inference-time scaling, architectures. Behind the open-source Granite models. Explicitly targets top-tier publications. Physics/math backgrounds historically well-regarded. Trustworthy/explainable AI track maps naturally to interpretability work.

### Salesforce AI Research — Research Scientist
- **Link:** https://careers.salesforce.com/en/jobs/jr214107/research-scientist-salesforce-ai-research/
- **Location:** Palo Alto CA, Seattle WA, San Francisco CA
- **Salary:** $117K–$344K depending on level and location
- **Status:** 🔲 Apply — role currently open (posted Feb 2026)
- **Notes:** Publishes seriously at top venues with real research autonomy. Focus areas: large-scale generative AI, agentic systems, multimodal deep learning, reasoning. More product-coupled than MSR or IBM but the research is genuine. Responsible AI / FATE track also open and maps to interpretability background.

---

## TIER 1C — ML + Broad Domain Science Roles

*Lead with the "parachute scientist" framing — rigorous ML brought to complex, partially-defined real-world problems across domains. Not biology specifically.*

### BCG X — AI Science Institute Postdoctoral Fellow
- **Link:** https://careers.bcg.com/global/en/job/57011/AI-Science-Institute-Postdoctoral-Fellow-United-States-BCG-X
- **Also on Indeed:** https://www.indeed.com/viewjob?jk=42e438ac868b6178
- **Location:** Boston, Seattle, or other US cities
- **Salary:** $158,400
- **Duration:** 24 months, with possible transition to full-time BCG X role
- **Status:** 🔲 Apply — HIGH PRIORITY
- **Notes:** Fellows expected to shape problem formulation and data strategy, not just run models. Focus areas span bioinformatics, climate, materials, computing, and more — not locked into one domain. Full access to BCG infrastructure, compute, and partner ecosystems, plus support for publishing.

### Argonne National Laboratory — Postdoctoral Appointee / Staff Scientist, AI/ML
- **Link:** https://www.anl.gov/hr/careers (search "machine learning" or "AI")
- **Location:** Lemont, IL (30 min from Chicago)
- **Status:** 🔲 Check open roles — rolling hiring
- **Notes:** Actively applying AI to materials science, climate, energy, national security — genuinely cross-domain. Home to Aurora supercomputer. ALCF and CELS directorate most relevant. Named fellowships: Margaret Butler (computational science) and Walter Massey.

### McKinsey QuantumBlack — Data Scientist / ML Scientist
- **Link:** https://www.mckinsey.com/capabilities/quantumblack/careers-and-community
- **Location:** Multiple US cities
- **Status:** 🔲 Check open roles
- **Notes:** Cross-domain ML across healthcare, energy, manufacturing, and more. **Caveat:** consulting-flavored — real time spent on client decks and KPIs. Worth it only if that tradeoff is acceptable.

---

## TIER 2 — Backup/Stepping Stone Programs

### Anthropic Fellows Program (May & July 2026 cohorts)
- **Link:** https://alignment.anthropic.com/2025/anthropic-fellows-program-2026/
- **Duration:** 4 months, paid
- **Notes:** Mechanistic interpretability is a listed focus area. Good backup if full-time role doesn't land.

### OpenAI Residency
- **Link:** https://openai.com/residency/
- **Status:** 2026 applications are closed. Monitor for future rounds.

### SPAR — AI Safety Research Fellowship
- **Link:** https://sparai.org/
- **Duration:** 3 months, part-time, remote
- **Notes:** Low commitment, good for building connections in the safety community.

---

## TIER 3 — Broader AI/Science Roles

### Google DeepMind, Science Team (AlphaFold/Protein Design)
- **Link:** https://deepmind.google/careers/
- **Notes:** Computational positions likely exist or will open. Relevant if open to biology-adjacent work.

### Basis Research Institute (Cambridge, MA)
- **Notes:** Research Scientist, ML/AI. Lists computational neuroscience and cognitive science as relevant PhD backgrounds.

### TikTok/ByteDance — ML Research Scientist, Atomistic AI (Seattle)
- **Notes:** AI for science at molecular scale. Check careers page.

---

## TIER 1B (continued)

### Perplexity — Research Internship
- **Status:** 🔲 Apply — details TBD
- **Notes:** Added 2026-03-17. Look up role and add link/details next session.

---

## Honorable Mentions (peruse when time allows)

- **FutureHouse** (futurehouse.org/fellowship) — AI-for-science nonprofit in SF. 2026 cohort likely closed; monitor for 2027.
- **Schmidt Futures — AI in Science Postdoctoral Fellowship** (schmidtfutures.org) — university-hosted, industry-funded, broad domain scope.
- **Allen Institute for AI (AI2)** — Seattle-based nonprofit. Serious ML research culture.
- **Sandia / Lawrence Berkeley / Oak Ridge National Labs** — similar to Argonne, different locations.
- **Flagship Pioneering / Recursion** — ML-for-biology; only relevant if appetite for bio-specific work returns.
