"""
Extended path patching — search for the candidates' causal effect through
other pathways before concluding they have none.

Tests:
  (1) MLPs allowed to recompute downstream — does the path run through MLPs?
  (2) Receivers = Backup NMs (8 heads) — backup pathway test.
  (3) Receivers = Negative NMs (L10H7, L11H10).
  (4) Receivers = ALL attention heads downstream of sender (permissive).
"""
from __future__ import annotations
from functools import partial
import torch

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer

from patching_experiments import build_prompt_pairs, logit_diff, DEVICE, N_PROMPTS

CANDIDATES = [(7, 11), (8, 1), (8, 11)]
NAME_MOVERS = [(9, 6), (9, 9), (10, 0)]
BACKUP_NMS = [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)]
NEGATIVE_NMS = [(10, 7), (11, 10)]
ALL_LATE_NMS = NAME_MOVERS + BACKUP_NMS + NEGATIVE_NMS

S_INHIBITION = [(7, 3), (8, 6), (8, 10)]    # known indirect movers for sanity


def make_hooks_path_patch(
        model, clean_cache, corr_cache, sender, receivers,
        freeze_mlps: bool, freeze_other_attn: bool):
    """Build hooks for path patching from sender to receivers.

    - sender's z is replaced with corrupted z (only that head).
    - if freeze_other_attn: all non-receiver attention layers are frozen at clean
      values. If False: all attention free to recompute except sender layer.
    - if freeze_mlps: all MLPs frozen at clean. If False: free to recompute.
    """
    sender_layer, sender_head = sender
    receiver_layers = {l for l, _ in receivers}

    def hook_z(z, hook, layer):
        if layer == sender_layer:
            new_z = clean_cache["z", layer].clone()
            new_z[:, :, sender_head, :] = corr_cache["z", layer][:, :, sender_head, :]
            return new_z
        if freeze_other_attn and layer not in receiver_layers:
            return clean_cache["z", layer]
        return z

    def hook_mlp(mlp_out, hook, layer):
        if freeze_mlps:
            return clean_cache["mlp_out", layer]
        return mlp_out

    hooks = []
    for l in range(model.cfg.n_layers):
        hooks.append((tl_utils.get_act_name("z", l), partial(hook_z, layer=l)))
        hooks.append((tl_utils.get_act_name("mlp_out", l), partial(hook_mlp, layer=l)))
    return hooks


def run_pp(model, hooks, clean_tokens, io_tok, s_tok):
    out = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
    ld = logit_diff(out, io_tok, s_tok)
    return ld.mean().item()


def main():
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    pairs = build_prompt_pairs(N_PROMPTS)
    clean_tokens = model.to_tokens([p["clean"] for p in pairs])
    corr_tokens = model.to_tokens([p["corrupted"] for p in pairs])
    io_tok = torch.tensor([model.to_single_token(" " + p["IO"]) for p in pairs], device=DEVICE)
    s_tok = torch.tensor([model.to_single_token(" " + p["S"]) for p in pairs], device=DEVICE)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corr_logits, corr_cache = model.run_with_cache(corr_tokens)
    baseline = logit_diff(clean_logits, io_tok, s_tok).mean().item()
    corrupted = logit_diff(corr_logits, io_tok, s_tok).mean().item()
    print(f"baseline clean     : {baseline:+.3f}")
    print(f"baseline corrupted : {corrupted:+.3f}\n")

    senders = CANDIDATES + S_INHIBITION
    receiver_sets = {
        "NMs": NAME_MOVERS,
        "Backup-NMs": BACKUP_NMS,
        "Neg-NMs": NEGATIVE_NMS,
        "All-late-NMs": ALL_LATE_NMS,
    }
    config_grid = [
        ("freeze_attn+MLP",      True,  True),
        ("freeze_attn, free_MLP", True, False),
        ("free_attn, freeze_MLP", False, True),
        ("free_all",              False, False),
    ]

    for cfg_name, freeze_other, freeze_mlps in config_grid:
        print("=" * 80)
        print(f"Config: {cfg_name}  (freeze_other_attn={freeze_other}, freeze_mlps={freeze_mlps})")
        print("=" * 80)
        for rcv_name, rcv in receiver_sets.items():
            print(f"\n  Receivers = {rcv_name}")
            for sender in senders:
                hooks = make_hooks_path_patch(
                    model, clean_cache, corr_cache, sender, rcv,
                    freeze_mlps=freeze_mlps, freeze_other_attn=freeze_other)
                ld = run_pp(model, hooks, clean_tokens, io_tok, s_tok)
                tag = "**cand**" if sender in CANDIDATES else " S-inhib"
                print(f"    {tag}  L{sender[0]}H{sender[1]:<2d}  "
                      f"logit_diff={ld:+.3f}  Δ={ld - baseline:+.3f}")
        print()


if __name__ == "__main__":
    main()
