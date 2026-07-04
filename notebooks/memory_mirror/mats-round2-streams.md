---
name: mats-round2-streams
description: MATS Autumn 2026 Round 2 — stream-by-stream fit assessment and application strategy for Peter
metadata: 
  node_type: memory
  type: project
  originSessionId: 14f8c394-1e56-45de-997d-4ff4de657c44
---

# MATS Autumn 2026 — Round 2 Stream Analysis

**Status:** Advanced to Round 2 on 2026-06-11. Deadline: **2026-06-23 11:59 PM AoE**.
**Format:** Choose streams to apply to (4 max realistically given time). Each stream has its own application questions of varying length.
**Source file:** `~/Downloads/[PUBLIC] 11.0 Streams for Stage 2.csv` (45 streams total)

---

## Stream rankings (best fit for Peter's profile)

### Tier 1 — strongest fits, apply with real effort

#### 1. Gary Abel (Fourth Eon Bio) — *strongest fit on the entire list*
- **Tracks:** Biosecurity, Empirical
- **Why it fits:** Mech interp applied to protein foundation models. Peter's dissertation was on energy-based models trained on protein sequences (bmDCA / Potts models / MSA-trained EBMs). He has BOTH molecular biophysics AND AI/ML expertise. Required skills checked: biomolecular sequence/structure/function understanding, molecular biophysics. Preferred skills checked: hands-on testing biological AI models (the PRR paper IS this), mech interp experience (his blog posts).
- **Application:** 1-hour DNA sequence screening exercise — sketch investigation approach for mystery sequence (100-200 words), write Python script implementing 1-2 steps using Biopython/BLAST, describe AI use + own judgment.
- **Peter's unique angle:** the rare overlap candidate — most applicants will be either bio people who don't know mech interp, or interp people who don't know biology. Peter is both.
- **Note:** Biosecurity flavor doesn't lock him into biosecurity permanently — work is mech interp of protein models, a research direction he could continue or pivot from.
- **✅ DONE — DNA-screening exercise COMPLETED & SUBMITTED as part of MATS R2 (finished in a later session; entire R2 application is in). The approach/script/result below were drafted in an earlier session (~2026-06-22).**
  - **Part 1 (approach, ~200 words):** translate to protein (defeats synonymous-codon obfuscation) → 6-frame translate (3 fwd + 3 rev-comp) → find ORF → blastp vs nr/UniProt (assess %id, coverage *over the functional region*, E-value, annotated function) → if inconclusive, Pfam/HMM profile search (catches diverged/embedded toxin domains BLAST misses; check conserved catalytic residues) → if still inconclusive, ESMFold → Foldseek vs PDB. Flag logic: confident, well-covered match to a *function of concern* with functional residues intact and no benign sequence-level explanation; false positives are costly so flag only with sufficient confidence.
  - **Part 2 (Biopython script):** `notebooks/other_jobs/mats/screen_sequence.py` — 6-frame translate + ORF + blastp via `Bio.Blast.NCBIWWW.qblast`. On the mystery 288-nt order: only frames +1 and −1 are stop-free (96 aa each, opposite strands); +1 is low-complexity (Asn/Ile/Phe-rich, weaker candidate), −1 is protein-like. **blastp of both vs Swiss-Prot and nr returned no homolog, even relaxed (low-complexity filter off, E≤2000 → only noise, best E≈220).** Conclusion: ORF is invisible to all sequence-homology search → possibly de-novo-designed or intentionally mutated → escalate to HMM/Pfam + structure, do NOT clear. (Also ran HMMER/InterProScan as a side-check during the session — zero domain matches — but the submitted write-up covers BLAST only.)
  - **AI-use note Peter wrote:** used Claude Code (CC) for a refresher on the bioinformatics tools (newly learned Foldseek) and to write/run the Python BLAST script; his own expertise = protein modeling (dissertation EBMs) + mech interp, plus the investigation strategy and flag/no-flag judgment.

#### 2. ARC — Alignment Research Center (Theory)
- **Mentors:** Jacob Hilton, Wilson Wu, Victor Lecomte, Michael Winer, Paul Christiano
- **Track:** Theory (only Theory-track stream he'd fit)
- **Why it fits:** Cumulant propagation = Peter's dissertation language. Heat capacity = Var(E) = second cumulant. The variance-about-data-mean criterion he developed in the PRR revision is literally a second-cumulant matching argument. "Mathematical maturity and a math, physics or computer science background" directly matches physics PhD. Mostly theoretical, no heavy PyTorch ask.
- **Application:** Short paragraph on prior alignment work / engagement with ARC's agenda.
- **Peter's angle:** Dissertation propagates κ₂(E) through different sampling regimes. Cumulant propagation as ARC develops it is the natural language. Wants to learn how to push to deeper networks + prove matching-sampling bounds.
- **Caveat (CORRECTED 2026-06-23 — supersedes "Why it fits"/"angle" above):** Peter confirmed he has NOT read ARC's work closely and does NOT understand cumulant propagation in ARC's sense or the "Competing with Sampling" post. The "cumulant propagation = his dissertation" framing was an unvalidated past-session brainstorm he never bought or internalized. **Do NOT have him claim engagement with ARC's agenda he doesn't have** — it would blow up in the work test / interview (same theory people). For the stream paragraph, answer ONLY the honest "prior alignment work" half (his IOI prompt-contrast interp). He already has the ARC work-test invite (general math/probability puzzles, needs zero ARC-agenda knowledge), so faking understanding gains nothing and risks a lot.

#### 3. Lee Sharkey (Goodfire AI)
- **Track:** Empirical (mech interp focused)
- **Why it fits:** Sharkey explicitly values "advanced math topics that have not yet been widely used in interpretability research" — direct invitation for physics PhDs. Parameter Decomposition method has geometric/topological flavor that connects to Peter's W_QK metric idea. Mech interp focused, aligns with his blog work.
- **Application:** 300-word research proposal on what he'd like to research.
- **Peter's angle:** Could propose extending parameter decomposition with metric structure from W_QK symmetric parts, or applying his prompt-contrast methodology to test PD-decomposed features.
- **Caveat:** Python/PyTorch engineering ask is real. Peter's Python is improving but still a gap.

### Tier 2 — apply if bandwidth allows

#### 4. OpenAI Safety Team
- **Mentors:** 19 mentors spanning interp, oversight, control, scheming, conceptual work
- **Track:** Empirical (with some conceptual angles)
- **Why it fits:** Broad mentor list includes "CoT and activation monitorability, representation-based interpretability" (compatible with Peter's forward-pass prompt-contrast methodology). "More conceptual work on model spec, automated AI research, and concentration of power... strong writing and conceptual analysis skills instead of empirical ML" — explicit non-engineer path that suits Peter.
- **Application:** Heavy — 500-900 word follow-up research proposal based on a specific OpenAI alignment post.
- **Peter's note (2026-06-17):** He is NOT against OpenAI, only was against cold-applying. Being in MATS Round 2 is different. Reconsidered and reinstated as a valid option.

#### 5. Anthropic
- **Mentors:** Bricken, Lindsey, Marks, Bowman, others (11 total)
- **Track:** Empirical
- **Why it fits:** Big tent including mech interp (Bricken's SAEs, Lindsey's mech interp, Marks's cognitive oversight). Peter's IOI / forward-pass work is in this universe.
- **Application:** Short (2 questions): Why interested in megastream + which safety research area excites you and what further work you'd want to do.
- **Caveat:** Recently rejected from Anthropic Fellows (2026-06-11) — signal is murky. Cheap to apply but uncertain.

#### 6. Apollo Research Science of Scheming
- **Mentors:** Teun van der Weij, Alexander Meinke
- **Track:** Empirical
- **Why it fits:** "Mathematical modelling (physics, computational neuroscience, etc) background" + "Philosophy background" as preferred attributes. Peter's stat-mech background + Catholic / symbol-grounding philosophy give two preferred-axis matches.
- **Application:** Scheming RL experiment design (~600 words) + project-decision process question.
- **Caveat:** Scheming/deception focus is unfamiliar ground for Peter. He'd be learning the literature. The experiment-design question is RL-flavored which is far from his wheelhouse.

### Skip
- **Team Shard (Alex Turner, Alex Cloud):** Yudkowskian/LessWrong DNA conflicts with Peter's stated anti-rationalist stance.
- **Google DeepMind, LawZero, MSL Deep Alignment (Meta):** Heavy engineering / PyTorch / RL training requirements that don't play to Peter's strengths.
- **Redwood Research, Daniel Kang, Maksym Andriushchenko:** Empirical streams with engineering / security focus not aligned.
- **All Biosecurity streams except Gary Abel:** Wrong domain. Gary Abel uniquely combines biosecurity with mech interp + proteins.
- **All Policy/Governance, Founding/Field-Building, Strategy/Forecasting:** Wrong domain.

---

## Strategic recommendation

**Apply to 3 streams with real effort, plus 1 cheap backup:**
1. **Gary Abel** — strongest fit, take the screening exercise seriously
2. **ARC** — short application, direct stat-mech connection (only if Peter is willing to engage with cumulant propagation seriously)
3. **Lee Sharkey** — 300-word proposal where his physics + mech interp can shine
4. **Anthropic** — cheap backup, 2 short questions

If Peter does ARC: lead with dissertation → cumulant propagation connection.
If Peter does Lee Sharkey: lead with parameter decomposition + W_QK metric / geometric angle.
If Peter does Gary Abel: lead with the rare-combo angle — biomolecular biophysics + mech interp + EBM/protein experience.

OpenAI is a substitute for one of the above if Peter has bandwidth — the 500-900 word proposal is the limiting factor.

---

## Key application notes per stream

| Stream | App length | Key skill ask | Peter's lead |
|--------|------------|---------------|--------------|
| Gary Abel | 1hr exercise | bio + Python + judgment | rare bio+interp overlap |
| ARC | 1 paragraph | math maturity | dissertation = cumulant work |
| Lee Sharkey | 300 words | advanced math + ML eng | W_QK metric proposal |
| OpenAI | 500-900 words | conceptual or empirical | forward-pass interp methodology |
| Anthropic | 2 short questions | broad fit | retinal ganglion analogy |
| Apollo | ~600 words | physics + philosophy + scheming | math modeling angle |

---

## Files / sources
- Round 2 stream CSV: `~/Downloads/[PUBLIC] 11.0 Streams for Stage 2.csv` (45 rows)
- Peter's profile: [user_profile.md](user_profile.md), [user_profile_faith_philosophy.md](user_profile_faith_philosophy.md)
- His research material to draw from: [research_ideas.md](research_ideas.md), [anthropic-fellows-app.md](anthropic-fellows-app.md), [astra-app.md](astra-app.md), [pivotal-app.md](pivotal-app.md), [bluedot-app.md](bluedot-app.md)
- The ARC blog post Peter shared in this session is at https://www.alignment.org/blog/competing-with-sampling/ (Eric Neyman et al., 2025-11-18)
