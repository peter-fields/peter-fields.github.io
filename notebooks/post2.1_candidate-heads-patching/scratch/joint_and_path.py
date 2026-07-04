"""
Two stronger causal tests for {L7H11, L8H1, L8H11}.

(A) Joint ablation: zero or patch all three candidates simultaneously. If they
    operate as a coordinated suppression module, backup compensation can't mask
    the effect the way it does for any single one.

(B) Path patching candidate → Name Mover (L9H6, L9H9, L10H0) input. Specifically
    patch the candidate's z with its corrupted-run version, freeze ALL other
    layer-8/9/10 outputs at their CLEAN values, and only let the downstream NMs
    (Q/K/V) recompute. Δ logit_diff isolates the candidate's *indirect* effect
    on the NMs — Wang et al.'s exact protocol for hidden circuit elements.
"""
from __future__ import annotations

from functools import partial
import numpy as np
import torch

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer

from patching_experiments import (
    build_prompt_pairs, logit_diff, DEVICE, N_PROMPTS,
)

CANDIDATES = [(7, 11), (8, 1), (8, 11)]
NAME_MOVERS = [(9, 6), (9, 9), (10, 0)]


def patch_specific_heads(z, hook, head_idxs, source_z):
    """Replace specified heads' z with `source_z` (same-shape slice)."""
    for h in head_idxs:
        z[:, :, h, :] = source_z[:, :, h, :]
    return z


def zero_specific_heads(z, hook, head_idxs):
    for h in head_idxs:
        z[:, :, h, :] = 0.0
    return z


# ---------------------------------------------------------------------------
# (B) Path patching helpers
# ---------------------------------------------------------------------------
# Path patching from sender (layer L_s, head h_s) to receiver Q/K/V at layer L_r:
#
#   1) Run clean. Cache everything.
#   2) Run corrupted. Cache sender's z only.
#   3) Re-run with hooks that:
#      - Freeze ALL attention heads at their clean cached values EXCEPT the sender:
#        replace sender's z with corrupted z.
#      - This perturbed residual stream flows into the receiver's QKV.
#      - At all layers >= receiver, freeze attention z at clean values (so only the
#        receiver actually sees the perturbed input via its QKV computation)
#      Then measure logit_diff change.
#
# We use a simpler equivalent: directly patch the sender; freeze MLPs and
# attention z at every other layer except the receivers — this means only the
# receivers respond to the sender perturbation. Their re-computed Q/K/V is what
# we want to measure.

def path_patch_sender_to_receivers(
        model, sender, receivers, clean_tokens, clean_cache, corr_cache,
        io_tok, s_tok):
    """Returns logit_diff after patching sender, allowing only receivers to recompute."""
    sender_layer, sender_head = sender
    receiver_layers = sorted({l for l, _ in receivers})

    def hook_freeze_attn_z(z, hook, layer):
        # Freeze all heads at clean except for sender head on its own layer
        if layer == sender_layer:
            # Patch only the sender head, freeze others
            new_z = clean_cache["z", layer].clone()
            new_z[:, :, sender_head, :] = corr_cache["z", layer][:, :, sender_head, :]
            return new_z
        elif layer in receiver_layers:
            # Receivers must RECOMPUTE — don't freeze their z. Just return as-is.
            return z
        else:
            # All other layers: freeze at clean
            return clean_cache["z", layer]

    def hook_freeze_mlp_out(mlp_out, hook, layer):
        # Freeze all MLPs at clean
        return clean_cache["mlp_out", layer]

    hooks = []
    for l in range(model.cfg.n_layers):
        hooks.append((tl_utils.get_act_name("z", l),
                      partial(hook_freeze_attn_z, layer=l)))
        hooks.append((tl_utils.get_act_name("mlp_out", l),
                      partial(hook_freeze_mlp_out, layer=l)))

    out = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
    return logit_diff(out, io_tok, s_tok)


def main():
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    pairs = build_prompt_pairs(N_PROMPTS)
    clean_tokens = model.to_tokens([p["clean"] for p in pairs])
    corr_tokens = model.to_tokens([p["corrupted"] for p in pairs])
    io_tok = torch.tensor(
        [model.to_single_token(" " + p["IO"]) for p in pairs], device=DEVICE)
    s_tok = torch.tensor(
        [model.to_single_token(" " + p["S"]) for p in pairs], device=DEVICE)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corr_logits, corr_cache = model.run_with_cache(corr_tokens)
    baseline = logit_diff(clean_logits, io_tok, s_tok).mean().item()
    corrupted = logit_diff(corr_logits, io_tok, s_tok).mean().item()

    print(f"baseline clean     : {baseline:+.3f}")
    print(f"baseline corrupted : {corrupted:+.3f}\n")

    # -----------------------------------------------------------------
    # (A) Joint ablation
    # -----------------------------------------------------------------
    print("=" * 70)
    print("(A) Joint ablations on the SAME forward pass")
    print("=" * 70)

    # Group hooks by layer for each set
    def run_zero_set(head_set, label):
        layers = sorted({l for l, _ in head_set})
        hooks = []
        for l in layers:
            heads_in_l = [h for ll, h in head_set if ll == l]
            hooks.append((tl_utils.get_act_name("z", l),
                          partial(zero_specific_heads, head_idxs=heads_in_l)))
        out = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
        ld = logit_diff(out, io_tok, s_tok).mean().item()
        print(f"  {label:42s}  LD={ld:+.3f}  Δ={ld - baseline:+.3f}  "
              f"%recov={(ld - corrupted) / (baseline - corrupted) * 100:6.1f}")
        return ld

    def run_patch_set(head_set, label):
        layers = sorted({l for l, _ in head_set})
        hooks = []
        for l in layers:
            hooks.append((tl_utils.get_act_name("z", l),
                          partial(patch_specific_heads,
                                  head_idxs=[h for ll, h in head_set if ll == l],
                                  source_z=corr_cache["z", l])))
        out = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
        ld = logit_diff(out, io_tok, s_tok).mean().item()
        print(f"  {label:42s}  LD={ld:+.3f}  Δ={ld - baseline:+.3f}  "
              f"%recov={(ld - corrupted) / (baseline - corrupted) * 100:6.1f}")
        return ld

    run_zero_set(CANDIDATES, "ZERO  {L7H11, L8H1, L8H11} joint")
    run_patch_set(CANDIDATES, "PATCH {L7H11, L8H1, L8H11} joint")
    run_zero_set(NAME_MOVERS, "ZERO  {L9H6, L9H9, L10H0} (NMs)")
    run_patch_set(NAME_MOVERS, "PATCH {L9H6, L9H9, L10H0} (NMs)")
    # All 6 together
    run_zero_set(CANDIDATES + NAME_MOVERS, "ZERO  candidates ∪ NMs")
    # Pairwise candidates
    run_zero_set([(7, 11), (8, 1)], "ZERO  {L7H11, L8H1}")
    run_zero_set([(7, 11), (8, 11)], "ZERO  {L7H11, L8H11}")
    run_zero_set([(8, 1), (8, 11)], "ZERO  {L8H1, L8H11}")

    # -----------------------------------------------------------------
    # (B) Path patching: candidate → Name Movers
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("(B) Path patching: sender (candidate) → receiver (NMs only recompute)")
    print("=" * 70)
    print("    Δ < 0 means: when sender is corrupted (other paths frozen at clean),")
    print("    NMs receive worse input and LD drops → sender is necessary INDIRECTLY.\n")

    # Sanity: path patch a known S-Inhibition head → NMs (should show effect)
    senders = [(7, 11), (8, 1), (8, 11), (7, 3), (8, 10), (8, 6), (10, 7), (0, 0), (1, 4)]
    for sender in senders:
        ld_pp = path_patch_sender_to_receivers(
            model, sender, NAME_MOVERS, clean_tokens, clean_cache, corr_cache,
            io_tok, s_tok)
        ld_mean = ld_pp.mean().item()
        delta = ld_mean - baseline
        recov = (ld_mean - corrupted) / (baseline - corrupted) * 100
        print(f"  sender L{sender[0]}H{sender[1]:<2d}  →  NMs  "
              f"LD={ld_mean:+.3f}  Δ={delta:+.3f}  %recov={recov:6.1f}")


if __name__ == "__main__":
    main()
