"""
Sanity check + direct logit attribution (DLA) for candidate heads.

DLA for a head = projection of that head's writes into the residual stream at the
final position onto the (W_U[IO] - W_U[S]) direction. This is the head's
contribution to logit_diff via the direct path (no routing through later layers).
It's the cleanest single-shot causal measure for the IOI task — what Wang et al.
used to flag Name Movers and Negative Name Movers.

Also runs zero-ablation on known Negative Name Movers (L10H7, L11H10) and a few
Name Movers as a baseline sanity check on the framework.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer

from patching_experiments import (
    build_prompt_pairs,
    logit_diff,
    hook_zero_head,
    DEVICE,
    N_PROMPTS,
)


SANITY_HEADS = [
    # Name Movers (Wang et al.) — ablating should decrease LD ideally
    (9, 6), (9, 9), (10, 0),
    # Negative Name Movers — ablating should INCREASE LD
    (10, 7), (11, 10),
    # S-Inhibition — ablating should decrease LD (no inhibition → S wins more)
    (7, 3), (8, 10),
    # Candidates
    (7, 11), (8, 1), (8, 11),
    # Negative controls
    (0, 0), (1, 4),
]


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
    print(f"baseline (clean)     : {baseline:+.3f}")
    print(f"baseline (corrupted) : {corrupted:+.3f}")

    # ---------------------------------------------------------------
    # Direct logit attribution per head
    # ---------------------------------------------------------------
    # head's contribution to residual at final pos = z[batch, -1, h, :] @ W_O[layer, h]
    # DLA = (head_contribution) @ (W_U[:, IO] - W_U[:, S]) per prompt
    # apply final LN to head_contribution before unembed to be precise

    W_U = model.W_U  # (d_model, vocab)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # logit_diff direction per prompt: (batch, d_model)
    ld_dir = W_U[:, io_tok] - W_U[:, s_tok]   # (d_model, batch)
    ld_dir = ld_dir.T                          # (batch, d_model)

    # Apply final layer norm scaling (we approximate by using the scale at final pos)
    # TransformerLens caches "ln_final.hook_scale" — apply that to head writes
    ln_scale = clean_cache["ln_final.hook_scale"][:, -1, :]   # (batch, 1)

    dla_clean = np.zeros((n_layers, n_heads))
    dla_corr = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        z_clean = clean_cache["z", layer][:, -1, :, :]   # (batch, H, d_head)
        z_corr = corr_cache["z", layer][:, -1, :, :]

        W_O = model.W_O[layer]   # (H, d_head, d_model)

        # Per-head contribution to residual at final position
        head_resid_clean = torch.einsum("bhd,hdm->bhm", z_clean, W_O)  # (batch, H, d_model)
        head_resid_corr = torch.einsum("bhd,hdm->bhm", z_corr, W_O)

        # Apply final LN scale
        head_resid_clean = head_resid_clean / ln_scale[:, None, :]
        head_resid_corr = head_resid_corr / ln_scale[:, None, :]

        # Project onto LD direction
        dla_c = torch.einsum("bhm,bm->bh", head_resid_clean, ld_dir).mean(0)
        dla_n = torch.einsum("bhm,bm->bh", head_resid_corr, ld_dir).mean(0)

        dla_clean[layer] = dla_c.cpu().numpy()
        dla_corr[layer] = dla_n.cpu().numpy()

    dla_diff = dla_clean - dla_corr

    # Save full DLA matrices
    OUT = Path(__file__).resolve().parent
    np.savez(OUT / "dla_results.npz",
             dla_clean=dla_clean, dla_corrupted=dla_corr, dla_diff=dla_diff,
             baseline=baseline, corrupted=corrupted)

    # Report on heads of interest + top movers
    print("\n=== Direct logit attribution (head → final-pos LD) ===")
    print(f"{'head':>8s}  {'DLA_clean':>10s}  {'DLA_corr':>10s}  {'Δ DLA':>10s}")
    for (l, h) in SANITY_HEADS:
        print(f"L{l}H{h:<2d}     "
              f"{dla_clean[l, h]:+10.3f}  "
              f"{dla_corr[l, h]:+10.3f}  "
              f"{dla_diff[l, h]:+10.3f}")

    print("\n=== Top 10 heads by Δ DLA (positive = contribute more to LD on clean) ===")
    flat = dla_diff.flatten()
    top = np.argsort(flat)[::-1][:10]
    for idx in top:
        l, h = divmod(idx, n_heads)
        print(f"  L{l}H{h:<2d}  Δ DLA = {dla_diff[l, h]:+.3f}  "
              f"(clean={dla_clean[l, h]:+.3f}, corr={dla_corr[l, h]:+.3f})")

    print("\n=== Bottom 10 heads by Δ DLA (negative = contribute MORE on corrupted) ===")
    bot = np.argsort(flat)[:10]
    for idx in bot:
        l, h = divmod(idx, n_heads)
        print(f"  L{l}H{h:<2d}  Δ DLA = {dla_diff[l, h]:+.3f}  "
              f"(clean={dla_clean[l, h]:+.3f}, corr={dla_corr[l, h]:+.3f})")

    # ---------------------------------------------------------------
    # Sanity check: zero-ablate known circuit heads
    # ---------------------------------------------------------------
    print("\n=== Zero-ablation on known IOI heads + candidates ===")
    print(f"{'head':>8s}  {'role':>22s}  {'LD':>8s}  {'Δ from clean':>14s}  {'%recov':>8s}")
    for (l, h) in SANITY_HEADS:
        out = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(tl_utils.get_act_name("z", l),
                        partial(hook_zero_head, head_idx=h))],
        )
        ld = logit_diff(out, io_tok, s_tok).mean().item()
        delta = ld - baseline
        recov = (ld - corrupted) / (baseline - corrupted) * 100 if baseline != corrupted else 0
        role = role_of(l, h)
        print(f"L{l}H{h:<2d}     {role:>22s}  {ld:+8.3f}  {delta:+14.3f}  {recov:8.1f}")


def role_of(layer, head):
    table = {
        (9, 6): "Name Mover", (9, 9): "Name Mover", (10, 0): "Name Mover",
        (10, 7): "Negative NM", (11, 10): "Negative NM",
        (7, 3): "S-Inhibition", (8, 10): "S-Inhibition",
        (7, 11): "**candidate**", (8, 1): "**candidate**", (8, 11): "**candidate**",
    }
    return table.get((layer, head), "non-circuit / control")


if __name__ == "__main__":
    main()
