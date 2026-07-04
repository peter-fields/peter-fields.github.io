"""
Where does the C_diff anti-correlation between candidates and Name Movers come from?

(1) Attention patterns: what do the candidates attend to on IOI vs non-IOI?
(2) Output norms and direction: how big is their write, and where does it point?
(3) Partial correlation: does the anti-correlation survive partialling out
    S-Inhibition heads (the obvious common upstream)?
(4) Per-position effect of ablating candidates: do downstream attention patterns
    (especially Name Movers) actually shift?
"""
from __future__ import annotations
from functools import partial
import numpy as np
import torch

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer

from patching_experiments import build_prompt_pairs, logit_diff, DEVICE, N_PROMPTS

CANDIDATES = [(7, 11), (8, 1), (8, 11)]
NAME_MOVERS = [(9, 6), (9, 9), (10, 0)]
S_INHIBITION = [(7, 3), (7, 9), (8, 6), (8, 10)]
N = 100  # bump up for cleaner correlations


def name_positions(model, prompt: str, names: dict) -> dict:
    """Token positions of named entities in `prompt`. Each name gets all positions
    where it tokenizes."""
    toks = model.to_tokens(prompt)[0].cpu().tolist()
    decoded = [model.to_string([t]) for t in toks]
    pos = {}
    for label, name in names.items():
        leading = " " + name
        positions = [i for i, d in enumerate(decoded) if d == leading]
        pos[label] = positions
    pos["final"] = [len(toks) - 1]
    return pos


def main():
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    pairs = build_prompt_pairs(N)
    clean_prompts = [p["clean"] for p in pairs]
    corr_prompts = [p["corrupted"] for p in pairs]
    clean_tokens = model.to_tokens(clean_prompts)
    corr_tokens = model.to_tokens(corr_prompts)
    io_tok = torch.tensor([model.to_single_token(" " + p["IO"]) for p in pairs], device=DEVICE)
    s_tok = torch.tensor([model.to_single_token(" " + p["S"]) for p in pairs], device=DEVICE)

    print(f"n_prompts = {N}")
    print(f"clean tokens: {tuple(clean_tokens.shape)}  corr tokens: {tuple(corr_tokens.shape)}\n")

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corr_logits, corr_cache = model.run_with_cache(corr_tokens)

    # ---------------------------------------------------------------
    # (1) Attention patterns at final position — where do candidates attend?
    # ---------------------------------------------------------------
    print("=" * 80)
    print("(1) Attention patterns at FINAL token position — per-name attention mass")
    print("=" * 80)

    def attention_mass_to_name(cache, tokens, prompts, layer, head, name_key, name_field):
        masses = []
        for i, p in enumerate(prompts):
            pos = name_positions(model, p, {"X": pairs[i][name_field]})
            attn = cache["pattern", layer][i, head, -1, :].cpu().numpy()
            masses.append(attn[pos["X"]].sum() if pos["X"] else 0.0)
        return np.array(masses)

    def attention_mass_bos(cache, layer, head):
        return cache["pattern", layer][:, head, -1, 0].cpu().numpy()

    summary_rows = []
    for layer, head in CANDIDATES + NAME_MOVERS + S_INHIBITION:
        row = {"head": f"L{layer}H{head}"}
        # On clean (IOI), report attention to IO, S, and BOS
        row["IOI: IO"] = attention_mass_to_name(clean_cache, clean_tokens, clean_prompts,
                                                 layer, head, "IO", "IO").mean()
        row["IOI: S"] = attention_mass_to_name(clean_cache, clean_tokens, clean_prompts,
                                                layer, head, "S", "S").mean()
        row["IOI: BOS"] = attention_mass_bos(clean_cache, layer, head).mean()
        row["nonIOI: IO"] = attention_mass_to_name(corr_cache, corr_tokens, corr_prompts,
                                                    layer, head, "IO", "IO").mean()
        row["nonIOI: S"] = attention_mass_to_name(corr_cache, corr_tokens, corr_prompts,
                                                   layer, head, "S", "S").mean()
        row["nonIOI: BOS"] = attention_mass_bos(corr_cache, layer, head).mean()
        summary_rows.append(row)

    cols = ["head", "IOI: IO", "IOI: S", "IOI: BOS", "nonIOI: IO", "nonIOI: S", "nonIOI: BOS"]
    print(f"{cols[0]:>8s}" + "".join(f"{c:>14s}" for c in cols[1:]))
    for row in summary_rows:
        line = f"{row['head']:>8s}"
        for c in cols[1:]:
            line += f"{row[c]:>14.3f}"
        print(line)

    # ---------------------------------------------------------------
    # (2) Per-head output norm at final position
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("(2) Output norm at final position: ||z_head @ W_O[layer, head, :, :]||")
    print("=" * 80)
    print(f"{'head':>8s}  {'||out||_IOI':>14s}  {'||out||_nonIOI':>16s}  {'ratio':>10s}")

    for layer, head in CANDIDATES + NAME_MOVERS + S_INHIBITION:
        z_clean = clean_cache["z", layer][:, -1, head, :]  # (batch, d_head)
        z_corr = corr_cache["z", layer][:, -1, head, :]
        W_O = model.W_O[layer, head]                        # (d_head, d_model)
        out_clean = z_clean @ W_O                            # (batch, d_model)
        out_corr = z_corr @ W_O
        n_clean = out_clean.norm(dim=-1).mean().item()
        n_corr = out_corr.norm(dim=-1).mean().item()
        print(f"L{layer}H{head:<2d}     {n_clean:>14.3f}  {n_corr:>16.3f}  {n_clean / max(n_corr, 1e-8):>10.3f}")

    # ---------------------------------------------------------------
    # (3) Projection onto a wider set of directions
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("(3) Projection of head output (at final pos) onto useful unembed directions")
    print("=" * 80)
    print("    avg per-prompt projection; positive = head writes in that direction")

    # Useful directions per prompt: W_U[:, IO], W_U[:, S], W_U[:, BOS]
    bos_tok = model.tokenizer.bos_token_id
    if bos_tok is None:
        bos_tok = 50256  # gpt2 default eos used as bos
    W_U = model.W_U
    dir_io = W_U[:, io_tok].T            # (batch, d_model)
    dir_s = W_U[:, s_tok].T
    dir_diff = dir_io - dir_s
    ln_scale = clean_cache["ln_final.hook_scale"][:, -1, :]   # (batch, 1)

    print(f"{'head':>8s}  {'<out, IO>':>12s}  {'<out, S>':>12s}  {'<out, IO-S>':>14s}  {'<out, ⟂>':>12s}")
    for layer, head in CANDIDATES + NAME_MOVERS + S_INHIBITION:
        z_clean = clean_cache["z", layer][:, -1, head, :]
        out_clean = (z_clean @ model.W_O[layer, head]) / ln_scale
        proj_io = (out_clean * dir_io).sum(dim=-1).mean().item()
        proj_s = (out_clean * dir_s).sum(dim=-1).mean().item()
        proj_diff = (out_clean * dir_diff).sum(dim=-1).mean().item()
        # Perpendicular component (norm of orthogonal complement to IO-S)
        diff_norm = dir_diff.norm(dim=-1, keepdim=True)
        diff_unit = dir_diff / (diff_norm + 1e-8)
        proj_along = ((out_clean * diff_unit).sum(dim=-1, keepdim=True)) * diff_unit
        perp = out_clean - proj_along
        perp_norm = perp.norm(dim=-1).mean().item()
        print(f"L{layer}H{head:<2d}     {proj_io:>12.3f}  {proj_s:>12.3f}  {proj_diff:>14.3f}  {perp_norm:>12.3f}")

    # ---------------------------------------------------------------
    # (4) Per-prompt KL diagnostics + correlations to confirm anti-correlation
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print(f"(4) Cross-head KL correlations on IOI prompts (n={N})")
    print("=" * 80)

    def per_prompt_kl(cache, tokens):
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        n_prompts = tokens.shape[0]
        seq_len = tokens.shape[1]
        log_n = np.log(seq_len)
        out = np.zeros((n_prompts, n_layers, n_heads))
        for L in range(n_layers):
            attn = cache["pattern", L]                    # (batch, H, dst, src)
            pi = attn[:, :, -1, :]                          # (batch, H, src)
            log_pi = torch.log(pi + 1e-12)
            H_pi = -(pi * log_pi).sum(dim=-1)               # (batch, H)
            kl = (log_n - H_pi.cpu().numpy()) / log_n
            out[:, L, :] = kl
        return out

    kl_ioi = per_prompt_kl(clean_cache, clean_tokens)
    kl_non = per_prompt_kl(corr_cache, corr_tokens)
    kl_ioi_flat = kl_ioi.reshape(N, -1)                  # (N, 144)
    kl_non_flat = kl_non.reshape(N, -1)

    def head_idx(layer, head):
        return layer * 12 + head

    def rho(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    print(f"\n{'pair':>30s}  {'r_IOI':>10s}  {'r_nonIOI':>10s}  {'Δr':>10s}")
    for c in CANDIDATES:
        for n in NAME_MOVERS:
            ci, ni = head_idx(*c), head_idx(*n)
            r_ioi = rho(kl_ioi_flat[:, ci], kl_ioi_flat[:, ni])
            r_non = rho(kl_non_flat[:, ci], kl_non_flat[:, ni])
            print(f"  L{c[0]}H{c[1]} ↔ L{n[0]}H{n[1]}                  "
                  f"{r_ioi:+10.3f}  {r_non:+10.3f}  {r_non - r_ioi:+10.3f}")

    # ---------------------------------------------------------------
    # (5) Partial correlation: partial out S-Inhibition heads
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("(5) Partial correlation candidate↔NM, controlling for S-Inhibition heads")
    print("=" * 80)
    print("    If partial r drops near zero, S-Inhibition is the common upstream cause.")

    def partial_corr(x, y, controls):
        """corr(x, y | controls) via residualization."""
        from numpy.linalg import lstsq
        # Residualize x and y against controls
        X = np.column_stack([np.ones(len(x)), controls])
        beta_x, *_ = lstsq(X, x, rcond=None)
        beta_y, *_ = lstsq(X, y, rcond=None)
        res_x = x - X @ beta_x
        res_y = y - X @ beta_y
        return float(np.corrcoef(res_x, res_y)[0, 1])

    si_idxs = [head_idx(*h) for h in S_INHIBITION]
    print(f"\n{'pair':>30s}  {'r_IOI':>10s}  {'partial r | S-inhib':>22s}  {'change':>10s}")
    for c in CANDIDATES:
        for n in NAME_MOVERS:
            ci, ni = head_idx(*c), head_idx(*n)
            controls = kl_ioi_flat[:, si_idxs]
            r = rho(kl_ioi_flat[:, ci], kl_ioi_flat[:, ni])
            pr = partial_corr(kl_ioi_flat[:, ci], kl_ioi_flat[:, ni], controls)
            print(f"  L{c[0]}H{c[1]} ↔ L{n[0]}H{n[1]}                  "
                  f"{r:+10.3f}  {pr:+22.3f}  {pr - r:+10.3f}")

    # ---------------------------------------------------------------
    # (6) Less-restrictive ablation: does ablating candidates change NM attention?
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("(6) Ablate ALL candidates; observe NM attention patterns at final pos")
    print("=" * 80)

    def zero_candidates_factory(layer):
        targets = [h for l, h in CANDIDATES if l == layer]
        def hook_fn(z, hook):
            for h in targets:
                z[:, :, h, :] = 0.0
            return z
        return hook_fn

    hooks = [(tl_utils.get_act_name("z", L), zero_candidates_factory(L))
             for L in sorted({l for l, _ in CANDIDATES})]

    # Use model.hooks() context to run_with_cache with active hooks
    with model.hooks(fwd_hooks=hooks):
        out_full, abl_cache = model.run_with_cache(clean_tokens)
    nm_changes = []
    for nl, nh in NAME_MOVERS:
        clean_attn = clean_cache["pattern", nl][:, nh, -1, :].cpu().numpy()  # (batch, src)
        abl_attn = abl_cache["pattern", nl][:, nh, -1, :].cpu().numpy()
        # Average attention to IO position vs S position vs BOS
        atts_clean = {"IO": [], "S": [], "BOS": []}
        atts_abl = {"IO": [], "S": [], "BOS": []}
        for i, p in enumerate(clean_prompts):
            pos_io = name_positions(model, p, {"IO": pairs[i]["IO"]})["IO"]
            pos_s = name_positions(model, p, {"S": pairs[i]["S"]})["S"]
            atts_clean["IO"].append(clean_attn[i, pos_io].sum() if pos_io else 0)
            atts_clean["S"].append(clean_attn[i, pos_s].sum() if pos_s else 0)
            atts_clean["BOS"].append(clean_attn[i, 0])
            atts_abl["IO"].append(abl_attn[i, pos_io].sum() if pos_io else 0)
            atts_abl["S"].append(abl_attn[i, pos_s].sum() if pos_s else 0)
            atts_abl["BOS"].append(abl_attn[i, 0])
        line = f"  L{nl}H{nh}: "
        for k in ["IO", "S", "BOS"]:
            mc = np.mean(atts_clean[k])
            ma = np.mean(atts_abl[k])
            line += f"  attn[{k}]: {mc:.3f} → {ma:.3f} (Δ={ma - mc:+.3f})"
        print(line)
    ld_abl = logit_diff(out_full, io_tok, s_tok).mean().item()
    print(f"\n  logit_diff with all candidates zeroed: {ld_abl:+.3f} "
          f"(baseline {logit_diff(clean_logits, io_tok, s_tok).mean().item():+.3f})")


if __name__ == "__main__":
    main()
