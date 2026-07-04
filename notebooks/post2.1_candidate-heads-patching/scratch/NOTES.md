# Post 2.1 — Investigation Notes

## Question
The C_diff correlation analysis from Post 2 flagged L7H11, L8H1, L8H11 as
candidate "missed" circuit elements based on a large anti-correlation flip with
core Name Movers (L9H6, L9H9):
  - r_nonIOI ≈ +0.5 to +0.7   (positively correlated when no name repeats)
  - r_IOI    ≈ -0.2 to -0.5   (anti-correlated when a name repeats)
  - Δr swings of +0.9 to +1.2 are among the largest in the model.

Question 1: where does the anti-correlation come from?
Question 2: do these heads play any causal role in IOI?

## Setup
- GPT-2 small. ABBA / BABA IOI templates (15 tokens each).
- Clean (IOI): "When Mary and John went to the store, John gave a drink to" → answer " Mary"
- Corrupted (ABC): same template, second {B} replaced by third name C — no repetition.
- Metric: logit_diff = logit(IO) − logit(S) at final position. Baseline ≈ +3.2, corrupted ≈ +0.8.
- n=50 for ablation runs, n=100 for correlation analysis.

## Findings — Question 2 (causal role)

### Direct logit attribution (head output projected onto IO−S at final pos)
| head            | DLA clean | DLA corr | Δ      |
|-----------------|-----------|----------|--------|
| L9H9   (NM)     | +2.19     | +0.41    | **+1.78** |
| L9H6   (NM)     | +1.11     | +0.30    | +0.81  |
| L10H7  (Neg NM) | -1.82     | -0.26    | **-1.57** |
| L8H10  (S-Inh)  | +0.30     | -0.02    | +0.33  |
| L7H3   (S-Inh)  | +0.11     | +0.01    | +0.10  |
| **L7H11** (cand)| +0.010    | +0.010   | **-0.000** |
| **L8H1**  (cand)| -0.003    | -0.002   | -0.001 |
| **L8H11** (cand)| -0.110    | -0.092   | -0.017 |

Candidates have essentially zero direct write into the IO−S direction. DLA framework
is calibrated correctly (NMs and Neg NMs at expected magnitudes).

### Output projections at final position (n=100, clean prompts)
| head    | ‖out‖ | <out, IO> | <out, S> | <out, IO-S> | <out, ⟂(IO-S)> |
|---------|------:|----------:|---------:|------------:|---------------:|
| L9H9    | 33.2  | +2.99     | +0.61    | **+2.38**   | 1.94           |
| L9H6    | 25.7  | +1.86     | +0.74    | +1.12       | 1.54           |
| L8H10   | 22.6  | -0.56     | -0.87    | +0.31       | 1.39           |
| L8H11   |  7.2  | +0.22     | +0.32    | **-0.10**   | 0.44           |
| L7H11   |  3.5  | +0.12     | +0.11    | +0.005      | 0.22           |
| L8H1    |  2.2  | +0.001    | +0.006   | -0.005      | 0.13           |

Candidates have small output norms (3.5, 2.2, 7.2 vs 16–33 for NMs). L8H11 actually
writes a small NEGATIVE IO−S component (towards S, opposite of NMs). L7H11 and L8H1
write ~zero in any name direction.

### Attention patterns at final position
| head    | IOI IO | IOI S | IOI BOS | non-IOI IO | non-IOI S | non-IOI BOS |
|---------|-------:|------:|--------:|-----------:|----------:|------------:|
| L9H9 (NM)   | 0.614 | 0.105 | 0.247 | 0.211 | 0.135 | 0.504 |
| L9H6 (NM)   | 0.546 | 0.177 | 0.269 | 0.214 | 0.149 | 0.468 |
| L8H10 (S-Inh)| 0.052 | 0.558 | 0.188 | 0.057 | 0.148 | 0.243 |
| **L7H11**   | 0.039 | 0.073 | **0.790** | 0.038 | 0.029 | 0.796 |
| **L8H1**    | 0.001 | 0.002 | **0.988** | 0.001 | 0.001 | 0.988 |
| **L8H11**   | 0.030 | 0.082 | **0.449** | 0.022 | 0.068 | 0.481 |

L7H11 and L8H1 are classic BOS-dumping no-op heads (79% / 99% attention to BOS).
L8H11 BOS-dumps 45%, spreads the rest. Differences between IOI and non-IOI are
small in absolute terms but shift consistently.

### Path patching (all configurations tested)
Sender = candidate, receivers ∈ {Name Movers, Backup NMs, Negative NMs, all late NMs}.
Tested with all 4 freeze configurations (attn freeze yes/no × MLP freeze yes/no).
  - L7H11 → any receiver set, any config: Δ logit_diff ∈ [+0.00, +0.05]
  - L8H1  → any receiver set, any config: Δ logit_diff ∈ [+0.00, +0.01]
  - L8H11 → any receiver set, any config: Δ logit_diff ∈ [+0.00, +0.02]
  - L7H3 (S-Inh) sanity:                  Δ ∈ [-0.43, -0.22]
  - L8H6 (S-Inh) sanity:                  Δ ∈ [-0.64, -0.27]
  - L8H10 (S-Inh) sanity:                 Δ ∈ [-1.27, -0.83]
  - L10H7 (Neg NM) sanity:                Δ = +1.57

Negative controls (L0H0, L1H4) sit at +0.006, +0.009 — same magnitude as candidates.

### Joint ablation
- Zero {L7H11, L8H1, L8H11}:    Δ logit_diff = −0.014
- Patch{L7H11, L8H1, L8H11}:    Δ logit_diff = +0.064
- Zero {NMs L9H6+L9H9+L10H0}:   Δ logit_diff = +0.395 (backup compensation; consistent with literature)
- Zero candidates ∪ NMs:        Δ logit_diff = +0.336

### Effect on NM attention when candidates are zeroed
- L9H6: attn[IO] 0.546 → 0.562 (+0.016)
- L9H9: attn[IO] 0.614 → 0.622 (+0.008)
- L10H0: attn[IO] 0.317 → 0.314 (-0.002)
- logit_diff: +3.297 → +3.277 (Δ −0.02)
NM attention is essentially unchanged.

**Verdict for Q2:** under any reasonable causal probe — direct effect, path through
NMs, path through Backup NMs, path through Negative NMs, joint ablation, with or
without MLPs frozen — the candidates produce changes within ±0.05 of baseline
logit_diff. Same magnitude as negative-control non-circuit heads. They do not play
a measurable causal role in IOI logit difference.

Caveats:
  - "No causal role in IOI logit_diff" is not "no role anywhere." Different task,
    different metric, different position, different prompt distribution — all
    untested.
  - Tiny effects (Δ ≈ +0.02 to +0.05) could in principle be real but below our
    statistical floor.

## Findings — Question 1 (origin of the anti-correlation)

### S-Inhibition heads as common upstream cause
Partial correlation candidate↔NM | S-Inhibition heads {L7H3, L7H9, L8H6, L8H10}:

| pair                | r_IOI | partial r | change | interpretation |
|---------------------|------:|----------:|-------:|----------------|
| L7H11 ↔ L9H6        | -0.44 | -0.43     | +0.01  | S-Inh does NOT explain |
| L7H11 ↔ L9H9        | -0.21 | -0.37     | -0.17  | anti-corr *strengthens* |
| L7H11 ↔ L10H0       | -0.40 | -0.19     | +0.21  | S-Inh partly explains |
| L8H1  ↔ L9H6        | -0.18 | -0.06     | +0.12  | S-Inh partly explains |
| L8H1  ↔ L9H9        | -0.30 | -0.17     | +0.12  | S-Inh partly explains |
| L8H1  ↔ L10H0       | -0.31 | **+0.17** | +0.48  | S-Inh fully explains + flips |
| L8H11 ↔ L9H6        | -0.27 | -0.13     | +0.14  | S-Inh partly explains |
| L8H11 ↔ L9H9        | -0.26 | -0.17     | +0.09  | S-Inh partly explains |
| L8H11 ↔ L10H0       | -0.53 | **-0.03** | +0.51  | S-Inh fully explains |

L8H1's anti-correlations are largely S-Inh-mediated. L8H11 ↔ L10H0 is fully
S-Inh-mediated. But L7H11 ↔ L9H6 / L9H9 anti-correlation is residual after S-Inh
control — not explained by S-Inhibition alone.

### Working hypothesis
The candidates' attention patterns shift between IOI and non-IOI because of
duplicate-token / S-Inhibition signal propagation, *but* their value-side writes
are too small to causally matter for logit_diff. The KL of their attention
distribution responds to the same upstream that drives NMs, just in opposite
direction (BOS-dumping decreases on IOI as duplicate signal arrives). This is the
core failure mode of C_diff: attention-pattern changes ≠ functional contribution.

Residual L7H11 ↔ L9 anti-correlation (not S-Inh-mediated) needs more probing —
candidates: Duplicate Token heads (L0H1, L3H0), Previous Token heads (L2H2, L4H11),
or earlier-layer features.

## Followup: Duplicate-Token heads as the dominant upstream cause

Successive partial correlation candidate↔NM, layering in upstream control sets:

| pair          | none  | +S-Inh | +S-Inh +DupTok | +S-Inh +DupTok +PrevTok | +all structural+Induction |
|---------------|------:|-------:|---------------:|------------------------:|--------------------------:|
| L7H11↔L9H6    | -0.44 | -0.43  | **-0.10**      | -0.35                   | -0.34                     |
| L7H11↔L9H9    | -0.21 | -0.37  | **+0.06**      | +0.07                   | +0.03                     |
| L7H11↔L10H0   | -0.40 | -0.19  | -0.07          | -0.19                   | -0.21                     |
| L8H1↔L9H6     | -0.18 | -0.06  | **+0.31**      | +0.26                   | +0.26                     |
| L8H1↔L9H9     | -0.30 | -0.17  | **+0.18**      | +0.04                   | -0.01                     |
| L8H1↔L10H0    | -0.31 | +0.17  | +0.38          | +0.26                   | +0.18                     |
| L8H11↔L9H6    | -0.27 | -0.13  | **+0.26**      | +0.26                   | +0.25                     |
| L8H11↔L9H9    | -0.26 | -0.17  | **+0.16**      | -0.04                   | -0.07                     |
| L8H11↔L10H0   | -0.53 | -0.03  | +0.16          | -0.06                   | -0.10                     |

**Adding the Duplicate Token heads (L0H1, L3H0) to the control set wipes out
or even flips the anti-correlation across 7 of 9 pairs.** Adding Previous Token
re-introduces some negative correlation (over-correction / multicollinearity),
but the dominant explanatory variable is the duplicate-token signal.

**Mechanistic story:** the duplicate-token signal is present on IOI prompts and
absent on non-IOI prompts. Both Name Movers and the candidate heads respond to
that signal (NMs become more peaked; candidates' BOS-dumping weakens slightly).
The opposite-direction responses in attention sharpness produce the C_diff
anti-correlation — without any direct functional link between them.

## Followup: Alternative output metrics under candidate ablation

| condition           | logit_diff | P(IO)  | P(S)   | rank(IO) | rank(S) |
|---------------------|-----------:|-------:|-------:|---------:|--------:|
| baseline            | +3.297     | 0.4537 | 0.0191 | 0.13     | 6.23    |
| candidates zeroed   | +3.277     | 0.4387 | 0.0187 | 0.16     | 6.24    |
| only L7H11 zeroed   | +3.148     | 0.4463 | 0.0214 | 0.12     | 5.78    |
| only L8H1  zeroed   | +3.281     | 0.4472 | 0.0190 | 0.11     | 6.30    |
| only L8H11 zeroed   | +3.474     | 0.4533 | 0.0163 | 0.12     | 6.84    |

Reading the individual ablations:
- **L7H11** zeroed → logit_diff drops -0.15 and S becomes slightly more probable
  (rank moves 6.23 → 5.78). Tiny *pro-IOI* effect — it weakly suppresses S.
- **L8H1** zeroed → effectively nothing. No-op head, full BOS-dump (99%).
- **L8H11** zeroed → logit_diff rises +0.18 and S becomes less probable (rank
  6.23 → 6.84). Tiny *anti-IOI* effect — it weakly boosts S.

L8H11's "anti-IOI" sign is consistent with its output projection: <out, S> > <out, IO>.
It writes mildly toward S (the wrong answer for IOI), so removing it helps.

## Per-token attention at final position (n=100, IOI)
| head   | BOS  | IO   | S    | other |
|--------|-----:|-----:|-----:|------:|
| L7H11  | 0.790 | 0.039 | 0.073 | 0.097 |
| L8H1   | 0.988 | 0.001 | 0.002 | 0.009 |
| L8H11  | 0.449 | 0.030 | 0.082 | 0.439 |

L8H11 spreads 44% of attention across non-name tokens — it's the most diffuse of
the three and the only one with substantive non-BOS, non-name routing.

## Revised picture

1. **L8H1 — true no-op.** Full BOS-dump, no causal effect, no role.
2. **L7H11 — weak distributed pro-IOI head.** Tiny effect (-0.15 logit_diff when
   ablated) in the same direction as Name Movers but ~10× smaller than even
   weak S-Inhibition heads.
3. **L8H11 — weak distributed anti-IOI head.** Tiny effect (+0.18 when ablated)
   in the same direction as Negative Name Movers, ~10× smaller than L10H7.

The C_diff anti-correlation is dominated by shared response to Duplicate-Token
upstream — that's the mechanism. There's a residual functional role for L7H11
and L8H11, weak and going in opposite directions, but it is real. L8H1 is noise.

This is more interesting than the binary "no role" verdict. It also shows
exactly where C_diff stops being trustworthy:
  - Adds task-distinguishing heads regardless of write-direction
  - Misses direction: L7H11 and L8H11 act oppositely but both register as
    "anti-correlated with NMs" because attention-pattern sharpness moves the
    same way under the shared duplicate-token drive.

## Next probes (if Peter wants)
- [ ] Project candidate outputs onto W_E[IO/S name] (embed side) — write to S2-position residual?
- [ ] Look at candidate behavior on non-final positions (S2, IO position residual contributions)
- [ ] Sweep all 144 heads for the same metric battery to see where L7H11/L8H11 sit
      among other "weak-but-real" contributors
- [ ] Try a fairer corruption (random-name-everywhere) and rerun ablation
- [ ] DLA-filter the original C_diff ranking: heads with task-distinguishing KL
      AND non-zero DLA on IO-S = better candidate list