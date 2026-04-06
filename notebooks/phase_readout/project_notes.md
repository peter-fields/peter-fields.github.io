# Phase Diagram of a Driven Izhikevich Neuron

**Project**: Peter Fields + Grace (mentee, physics background)
**Started**: 2026-03-31
**Reference**: Brunel 2000 — *Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons*

---

## Big Picture

Brunel 2000 characterizes a *network's* collective dynamics via a phase diagram (external drive vs. inhibition strength). The phases are:
- **SR** — Synchronous Regular: neurons fire together, periodically
- **AR** — Asynchronous Regular: stationary global rate, quasi-regular individual firing
- **AI** — Asynchronous Irregular: stationary global rate, strongly irregular individual firing
- **SI** — Synchronous Irregular: oscillatory global activity, irregular individual firing

We adapt this for a **single Izhikevich neuron** driven by a structured oscillatory input. The question: across (input strength, noise), when does the neuron faithfully follow the input oscillation vs. fire chaotically vs. go silent?

Our four target phases:
- **Silent**: neuron never reaches threshold
- **Synchronous**: neuron fires phase-locked to input oscillation
- **Asynchronous**: neuron fires at a steady rate uncorrelated with input
- **Disordered**: neuron fires chaotically / irregularly with no consistent phase relationship

---

## Model

### Izhikevich Neuron
```
dv/dt = 0.04*v^2 + 5*v + 140 - u + I(t)
du/dt = a*(b*v - u)
if v >= 30 mV:  v <- c,  u <- u + d
```
Parameters for regular spiking: `a=0.02, b=0.2, c=-65, d=8`

### Input Current
I(t) = A * x(t)

where `A` = input strength (phase diagram axis 1) and x(t) is the output of a **stochastic** damped harmonic oscillator:

```
dx/dt = y
dy/dt = -2*gamma*omega_0*y - omega_0^2*x + sigma*xi(t)
```

- `sigma` = noise amplitude driving the oscillator (phase diagram axis 2)
- `xi(t)` = Gaussian white noise — in Euler integration: sigma * sqrt(dt) * randn() per step
- Noise enters only through the oscillator, not directly into the neuron

The SDHO sustains itself indefinitely via the noise forcing — no re-kicking needed.

Suggested defaults: `omega_0 = 2*pi*10` (10 Hz), `gamma = 0.1` (lightly damped), `dt = 0.1` ms, x(0) = 0, y(0) = 0

---

## Order Parameters

Two observables computed per (A, sigma) grid point from the spike times:

**R — vector strength** (phase locking to input)
- For each spike time t_k, compute phase: phi_k = 2*pi * (t_k mod T_input) / T_input, where T_input = 2*pi / omega_0
- R = | (1/N) * sum_k exp(i * phi_k) |
- R = 1: perfect locking. R = 0: uniform phase distribution. NaN if no spikes.

**CV — coefficient of variation of ISI** (regularity of firing)
- ISI_k = t_{k+1} - t_k
- CV = std(ISI) / mean(ISI)
- CV ~ 0: regular. CV ~ 1: Poisson. CV > 1: bursty/disordered. NaN if < 2 spikes.

| Phase | R | CV |
|---|---|---|
| Silent | NaN | NaN |
| Synchronous | high | low |
| Asynchronous | low | moderate (~0.5-1) |
| Disordered | low | high (>1) |

R alone cannot separate asynchronous from disordered — both have R ~ 0. CV is needed to make that distinction. Plot both as separate imshow panels.

**Future**: spectral coherence C(f) = |S_xn(f)|^2 / (S_xx(f) * S_nn(f)) as a linear MI lower bound. Connects R (coherence at omega_0) to full information-theoretic analysis.

---

## Phase Diagram Plan

Axes:
- x-axis: `A` (input amplitude), e.g. 0 to 20
- y-axis: `sigma` (noise), e.g. 0 to 15

Grid: start 20x20, refine to 50x50 later.

For each (A, sigma):
1. Simulate T = 3000 ms total (dt = 0.1 ms)
2. Discard first 500 ms as transient
3. Collect spike times from remaining 2500 ms
4. Compute R and CV

Plot R and CV as two side-by-side imshow panels. NaN (silent) cells in grey.

---

## Milestones for Grace

1. [ ] Implement Izhikevich neuron, verify it fires correctly with constant input
2. [ ] Implement SDHO, verify it oscillates with noise sustaining it
3. [ ] Combine: drive neuron with A*x(t), confirm spikes appear
4. [ ] Implement R and CV from spike times
5. [ ] Sweep (A, sigma) grid, compute R and CV at each point
6. [ ] Plot two-panel phase diagram, identify four phases

---

## MI Calculation (not yet discussed with Grace)

The scientific claim is that MI between spike train and oscillator is maximized at the synchronous/asynchronous phase boundary. Two approaches:

**Option 1 — Coherence (cheap, do first)**
- Coherence: C(f) = |S_xn(f)|^2 / (S_xx(f) * S_nn(f))
- Linear MI lower bound: -sum_f log(1 - C(f)) * df
- One long simulation per grid point. Connects directly to R (which is coherence evaluated at omega_0).
- If coherence ridge aligns with phase boundary, that's already a clean result.

**Option 2 — Strong et al. 1998 (full MI)**
- MI = H(spikes) - H(spikes | stimulus)
- Requires many repeats of the *same* x(t) trajectory (fix the noise seed, rerun neuron many times)
- Expensive but gives full MI including nonlinear contributions
- Do this only if coherence result is insufficiently convincing

**The hypothesis**: MI ridge tracks the R/CV phase boundary as Izhikevich parameters vary. Phase boundary location is a parameter-free predictor of optimal readout.

## Origin and Longer-Term Vision

Original motivation (Peter + Cheyne): non-reciprocally coupled populations of neurons show interesting avalanching transitions between phases (e.g. silent to synchronous swaps between populations), and the hypothesis was that these transitions are optimal for downstream readout.

Cheyne claims there is a formal mathematical mapping between the two systems such that the phase diagrams are topologically identical — with the x-axis relabeled from input amplitude A (single neuron) to non-reciprocal coupling strength J_AB - J_BA (network). Non-reciprocity controls avalanching dynamics the same way A controls phase-locking in the single neuron. **This mapping is Cheyne's claim and should be verified before building on it.**

The avalanching dynamics in the network are likely the network-level signature of the same critical sensitivity that maximizes MI at the single neuron phase boundary. At the subcritical/supercritical transition (controlled by non-reciprocal coupling strength), activity propagates "just the right amount" — neither dying out nor exploding — which is the mechanistic explanation for maximal input sensitivity and maximal MI.

If the single neuron result holds, the non-reciprocal population system is a natural follow-up: do the avalanching transitions between populations also sit at MI maxima, with non-reciprocal coupling strength playing the role of A? That could be a second paper.

## Open Questions

- Does the MI ridge align tightly with the phase boundary, or is it broad?
- Does the alignment hold as d (adaptation strength) is varied?
- Is the single Izhikevich neuron mathematically related to non-reciprocally coupled populations? (Cheyne's claim — unverified)
- Is a single neuron the right model, or do we want a small population?

---

## Files

- `project_notes.md` — this file
- `A_1008925309027.pdf` — Brunel 2000
- `instructions_grace.md` / `instructions_grace.pdf` — Grace's task 1 instructions
