---
name: idea_alternating_attention
description: Research idea — alternating causal/bidirectional attention for retroactive recontextualization
type: project
---

## Alternating Causal/Bidirectional Attention

**Core idea**: Causal transformers can only do *anticipation* — every token is processed given what came before. But meaning often works through *retroactive recontextualization* — earlier tokens need to be reinterpreted in light of later ones. Plot twists, betrayal reveals, "I love you" after a long buildup. Causal architecture structurally cannot do this; it can only have learned to anticipate the recontextualization before it happens.

**Proposed mechanism** (refined): Single weight set. Generate X tokens causally with KV cache. Every X tokens:
1. Run full bidirectional pass (no causal mask) over full sequence — updates all residual streams and KV cache for tokens 1...T-1
2. Run one causal forward pass for token T only, attending to the updated bidirectional KV cache
3. Generate T+1 from T's logits. Continue causally.

Do NOT recompute causal KV after the bidirectional pass — that would undo the retroactive updates. The bidirectional KV *is* the new cache. Token T's causal pass attends to richer keys that incorporated future context.

**Cost**: One bidirectional pass + one single-token causal pass every X tokens. Much cheaper than originally thought — no full causal recompute needed.

**Training requirement**: Must train with this alternating pattern from scratch — weights need to have seen bidirectionally-updated residual streams during training or the re-encoding is out-of-distribution and useless.

**Sharp training angle**: Identify which tokens in a corpus *require* backward context to be correctly interpreted — tokens whose meaning shifts substantially given later information. This is a learnable signal. Could:
- Tell you where the architecture actually matters vs. wasted compute
- Suggest a curriculum: train mostly causal, fine-tune bidirectional passes on high-recontextualization examples specifically
- Potentially remove redundancy in training protocol

**Framing**: This *removes* a constraint (the causal mask, periodically) rather than adding bells and whistles. Makes the model more faithful to what transformers already are.

**Experimental sequence** (each step conditional on previous):
1. Does bidirectional refresh help at all? — existence proof, ignore efficiency. Pick one task where retroactive recontextualization is clearly required (coreference with post-hoc disambiguation, or synthetic plot-twist data). Run unlimited refreshes. Show improvement over causal baseline.
2. How much does refresh frequency matter? — find the knee in the performance vs. refresh-rate curve.
3. Can a smaller refresh model match a larger causal model? — parameter efficiency claim: "same performance, fewer parameters." Only worth asking if step 1 is non-null.

**Deeper efficiency argument**: Causal models compensate for the architectural limitation by storing anticipatory statistics — hedging every ambiguity because they can't resolve it retroactively. That takes capacity. A model that can revise earlier representations doesn't need to hedge — it can commit, then update. Intermittent bidirectional refresh may be a compute-efficient inductive bias: not more parameters, but better use of the ones you have.

**Framing / name**: "Dynamic self-distillation via intermittent bidirectional passes" — the model teaches itself better representations on the fly. Bidirectional pass = online teacher; causal generation = student. No separate model, just a mask change. Connects to Hinton-style knowledge distillation but internal and dynamic.

**Open question — circuit structure under dual-mode weights**: W_QK and W_OV now serve two jobs. In causal mode: skip-trigram implementer, always attends backward. In bidirectional mode: full-context integrator, attends anywhere. These are genuinely different functions. Do the learned weights factorize cleanly between modes, or does the model find dual-purpose matrices that do both reasonably well? Optimization pressure from both objectives might conflict, or might find a surprisingly clean solution. Interpretability angle: circuit analysis on this model would look very different from Elhage — heads have mode-dependent behavior, and the same W_QK implements different "which token to attend to" logic depending on whether the mask is on or off.

**Minimal probe experiment — residual stream update check**:
Before training anything, check whether the bidirectional pass actually updates early token representations meaningfully on a pretrained model (e.g. GPT-2, out-of-distribution but cheap).
- Construct minimal pairs: "John said he was a doctor. [trustworthy continuation]" vs "John said he was a doctor. [liar reveal continuation]"
- Run causal forward pass, record John's residual stream
- Run bidirectional pass on the same sequence, record John's residual stream again
- Define a "liar" direction: mean difference in residual streams between liar vs trustworthy contexts (no probe training needed)
- Measure cosine similarity of the delta (post - pre bidirectional) with the liar direction
- If the delta aligns with the liar direction in liar-reveal sequences but not trustworthy ones → bidirectional pass is doing retroactive recontextualization even on causally-trained weights
- If no signal → weights are too out-of-distribution; need fine-tuning before the experiment is meaningful

This is the cheapest possible existence check. No training, no new architecture — just mask off, measure, mask on.

**Pilot experiment — IOI circuit**:
IOI (Indirect Object Identification, Wang et al. 2022) is ideal: circuit is fully characterized causally, so any change is interpretable.
1. Train small causal model on IOI data, reproduce known circuit (name-mover heads, etc.)
2. Init bidirectional model from those weights, continue training with intermittent refresh
3. Watch: does loss drop? Do W_QK / W_OV change substantially? Do circuit roles shift?
4. Then shrink the bidirectional model head-by-head and track degradation curve vs. causal baseline
- If bidirectional model holds IOI performance at smaller size → architectural capability substitutes for parameters
- If matrices barely move → causal model already found the optimal solution, bidirectionality adds nothing here
- IOI is forward-determined (answer known before output position), so if bidirectionality helps even here, that's stronger than expected

**Status**: Hand-wavy, worthy of a note. Not a current priority. Check for prior work before treating as novel — nothing known as of 2026-03-13.

**Closest prior work**: Diffusion LMs (MDLM, SEDD), XLNet, Insertion Transformer, Retro. None are exactly this.
