"""
Probe the remaining puzzle: residual L7H11 ↔ L9 anti-correlation that isn't
explained by S-Inhibition, and whether the candidates affect any metric beyond
logit_diff(IO-S).

(A) Partial out more upstream heads: Duplicate Token, Previous Token, Induction.
(B) Effect of candidate ablation on top-1 probability, IO probability mass, and
    next-token entropy.
(C) What does L7H11 attend to on IOI when NOT BOS-dumping? Per-position
    breakdown of attention at the final token for IOI vs non-IOI.
"""
from __future__ import annotations
from functools import partial
import numpy as np
import torch

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer

from patching_experiments import build_prompt_pairs, logit_diff, DEVICE

CANDIDATES = [(7, 11), (8, 1), (8, 11)]
NAME_MOVERS = [(9, 6), (9, 9), (10, 0)]
S_INHIBITION = [(7, 3), (7, 9), (8, 6), (8, 10)]
DUPLICATE_TOKEN = [(0, 1), (3, 0)]
PREVIOUS_TOKEN = [(2, 2), (4, 11)]
INDUCTION = [(5, 5), (6, 9)]
N = 100


def head_idx(layer, head):
    return layer * 12 + head


def rho(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(x, y, controls):
    from numpy.linalg import lstsq
    if controls.ndim == 1:
        controls = controls[:, None]
    X = np.column_stack([np.ones(len(x)), controls])
    beta_x, *_ = lstsq(X, x, rcond=None)
    beta_y, *_ = lstsq(X, y, rcond=None)
    return float(np.corrcoef(x - X @ beta_x, y - X @ beta_y)[0, 1])


def per_prompt_kl(model, cache, tokens):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    seq_len = tokens.shape[1]
    log_n = np.log(seq_len)
    out = np.zeros((tokens.shape[0], n_layers, n_heads))
    for L in range(n_layers):
        attn = cache["pattern", L][:, :, -1, :]
        log_pi = torch.log(attn + 1e-12)
        H_pi = -(attn * log_pi).sum(dim=-1)
        out[:, L, :] = ((log_n - H_pi.cpu().numpy()) / log_n)
    return out


def main():
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    pairs = build_prompt_pairs(N)
    clean_prompts = [p["clean"] for p in pairs]
    clean_tokens = model.to_tokens(clean_prompts)
    io_tok = torch.tensor([model.to_single_token(" " + p["IO"]) for p in pairs], device=DEVICE)
    s_tok = torch.tensor([model.to_single_token(" " + p["S"]) for p in pairs], device=DEVICE)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    baseline = logit_diff(clean_logits, io_tok, s_tok).mean().item()
    kl_ioi = per_prompt_kl(model, clean_cache, clean_tokens).reshape(N, -1)

    # ---------------------------------------------------------------
    # (A) Successive partial correlations
    # ---------------------------------------------------------------
    print("=" * 80)
    print("(A) Partial correlations: control for successively larger upstream sets")
    print("=" * 80)

    control_sets = {
        "none":                              [],
        "+ S-Inhibition":                    S_INHIBITION,
        "+ S-Inh + DuplicateToken":          S_INHIBITION + DUPLICATE_TOKEN,
        "+ S-Inh + DupTok + PrevTok":        S_INHIBITION + DUPLICATE_TOKEN + PREVIOUS_TOKEN,
        "+ all structural + Induction":      S_INHIBITION + DUPLICATE_TOKEN + PREVIOUS_TOKEN + INDUCTION,
    }

    targets = [(c, n) for c in CANDIDATES for n in NAME_MOVERS]
    headers = list(control_sets.keys())
    print(f"\n{'pair':>14s}" + "".join(f"{h:>32s}" for h in headers))
    for (c, n) in targets:
        ci, ni = head_idx(*c), head_idx(*n)
        row = f"L{c[0]}H{c[1]}↔L{n[0]}H{n[1]:<2d}"
        for name, ctrl in control_sets.items():
            if not ctrl:
                r = rho(kl_ioi[:, ci], kl_ioi[:, ni])
            else:
                ctrl_idxs = [head_idx(*h) for h in ctrl]
                r = partial_corr(kl_ioi[:, ci], kl_ioi[:, ni], kl_ioi[:, ctrl_idxs])
            row += f"{r:>32.3f}"
        print(f"{row:<14s}")

    # ---------------------------------------------------------------
    # (B) Alternative metrics under candidate ablation
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("(B) Alternative metrics under candidate ablation")
    print("=" * 80)

    def metrics(logits, label):
        last = logits[:, -1, :]
        probs = torch.softmax(last, dim=-1)
        p_io = probs.gather(-1, io_tok[:, None]).squeeze(-1)
        p_s = probs.gather(-1, s_tok[:, None]).squeeze(-1)
        # entropy
        log_probs = torch.log_softmax(last, dim=-1)
        ent = -(probs * log_probs).sum(dim=-1)
        # rank of IO among all tokens
        top_probs, top_idx = probs.topk(5, dim=-1)
        io_in_top1 = (top_idx[:, 0] == io_tok).float().mean().item()
        io_in_top5 = (top_idx == io_tok[:, None]).any(dim=-1).float().mean().item()
        # rank
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        rank_io = (sorted_idx == io_tok[:, None]).int().argmax(dim=-1).float().mean().item()
        rank_s = (sorted_idx == s_tok[:, None]).int().argmax(dim=-1).float().mean().item()

        ld = logit_diff(logits, io_tok, s_tok).mean().item()
        print(f"  {label:30s}  logit_diff={ld:+.3f}  "
              f"P(IO)={p_io.mean().item():.4f}  P(S)={p_s.mean().item():.4f}  "
              f"H={ent.mean().item():.3f}  "
              f"rank(IO)={rank_io:.2f}  rank(S)={rank_s:.2f}  "
              f"P(top1=IO)={io_in_top1:.2f}  P(IO∈top5)={io_in_top5:.2f}")

    metrics(clean_logits, "baseline clean")

    # Zero all candidates
    def zero_candidates_factory(layer):
        targets = [h for l, h in CANDIDATES if l == layer]
        def hook_fn(z, hook):
            for h in targets:
                z[:, :, h, :] = 0.0
            return z
        return hook_fn

    hooks = [(tl_utils.get_act_name("z", L), zero_candidates_factory(L))
             for L in sorted({l for l, _ in CANDIDATES})]
    out = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
    metrics(out, "candidates zeroed")

    # Each candidate individually
    for c in CANDIDATES:
        L, H = c
        def factory(target_h):
            def hf(z, hook):
                z[:, :, target_h, :] = 0.0
                return z
            return hf
        out = model.run_with_hooks(
            clean_tokens, fwd_hooks=[(tl_utils.get_act_name("z", L), factory(H))])
        metrics(out, f"only L{L}H{H} zeroed")

    # ---------------------------------------------------------------
    # (C) L7H11 attention at final position — full breakdown by token role
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("(C) L7H11 attention at final position — per-token-role breakdown")
    print("=" * 80)

    def role_of_position(tok_decoded, prompt_meta):
        if tok_decoded == " " + prompt_meta["IO"]:
            return "IO"
        if tok_decoded == " " + prompt_meta["S"]:
            return "S"
        return "other"

    # For each candidate, mean attention to each token role at final pos
    for cand in CANDIDATES:
        L, H = cand
        attns = clean_cache["pattern", L][:, H, -1, :].cpu().numpy()   # (batch, src)
        # Categorize each token in each prompt
        roles = []
        for i, p in enumerate(clean_prompts):
            toks = clean_tokens[i].cpu().tolist()
            decoded = [model.to_string([t]) for t in toks]
            meta = pairs[i]
            role_seq = []
            for j, d in enumerate(decoded):
                if j == 0:
                    role_seq.append("BOS")
                elif d == " " + meta["IO"]:
                    role_seq.append("IO")
                elif d == " " + meta["S"]:
                    role_seq.append("S")
                else:
                    role_seq.append("other")
            roles.append(role_seq)
        roles = np.array(roles)  # (batch, seq)

        # Sum attention by role
        role_sums = {"BOS": [], "IO": [], "S": [], "other": []}
        for i in range(N):
            for r in role_sums:
                role_sums[r].append(attns[i, roles[i] == r].sum())
        print(f"  L{L}H{H}:  "
              f"BOS={np.mean(role_sums['BOS']):.3f}  "
              f"IO={np.mean(role_sums['IO']):.3f}  "
              f"S={np.mean(role_sums['S']):.3f}  "
              f"other={np.mean(role_sums['other']):.3f}  "
              f"(sum={np.mean(role_sums['BOS']) + np.mean(role_sums['IO']) + np.mean(role_sums['S']) + np.mean(role_sums['other']):.3f})")

    # ---------------------------------------------------------------
    # (D) L7H11 attention at non-final positions
    # ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("(D) L7H11 attention from every source token, on a single representative prompt")
    print("=" * 80)
    sample_idx = 0
    print(f"  prompt: {clean_prompts[sample_idx]}")
    sample_toks = clean_tokens[sample_idx].cpu().tolist()
    decoded = [model.to_string([t]) for t in sample_toks]
    for cand in CANDIDATES:
        L, H = cand
        attn = clean_cache["pattern", L][sample_idx, H].cpu().numpy()    # (dst, src)
        print(f"\n  L{L}H{H} attention matrix (dst → src), each row sums to 1:")
        print("    " + " ".join(f"{d!r:>10s}" for d in decoded))
        for dst_i, dst_d in enumerate(decoded):
            row = "    "
            for src_i in range(len(decoded)):
                if src_i > dst_i:
                    cell = "          "
                else:
                    v = attn[dst_i, src_i]
                    cell = f"{v:10.3f}" if v >= 0.005 else "         ."
                row += cell + " "
            print(f"  {dst_d!r:>10s} | " + row)


if __name__ == "__main__":
    main()