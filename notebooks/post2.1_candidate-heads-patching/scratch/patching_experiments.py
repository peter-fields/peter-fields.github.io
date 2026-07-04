"""
Post 2.1 — Causal verification of candidate IOI heads (L7H11, L8H1, L8H11).

Method: three complementary causal interventions on GPT-2-small.

  1. Zero ablation        — set head's z to zero.
  2. Mean ablation        — replace head's z with batch mean (over clean IOI prompts).
  3. Activation patching  — replace head's z on clean run with z from ABC-corrupted run
                            ("noising" direction: does the head matter on clean prompts?).

Metric: logit_diff = logit(IO) - logit(S) at the final token position.

Controls:
  - Positive: L9H6, L9H9 (core Name Movers from Wang et al.)
  - Negative: L0H0, L1H4 (early, non-circuit)

All ablations are head-level (z slice, NOT mlp).
"""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch

import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
N_PROMPTS = 50
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CANDIDATES = [(7, 11), (8, 1), (8, 11)]
POSITIVE_CONTROLS = [(9, 6), (9, 9)]            # Name Movers
NEGATIVE_CONTROLS = [(0, 0), (1, 4)]             # early, non-circuit
ALL_HEADS = CANDIDATES + POSITIVE_CONTROLS + NEGATIVE_CONTROLS

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "patching_results.json"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

TEMPLATES_ABBA = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "When {A} and {B} went to the park, {B} handed a ball to",
    "When {A} and {B} arrived at the office, {B} passed a note to",
    "When {A} and {B} got to the restaurant, {B} offered a menu to",
    "When {A} and {B} walked into the room, {B} showed a book to",
    "After {A} and {B} met at the cafe, {B} sent a message to",
    "After {A} and {B} sat down for dinner, {B} gave a gift to",
]
TEMPLATES_BABA = [t.replace("{A}", "X").replace("{B}", "{A}").replace("X", "{B}")
                  for t in TEMPLATES_ABBA]

NAMES = ["Mary", "John", "Alice", "Bob", "Sarah", "Tom",
         "Emma", "James", "Lisa", "David", "Kate", "Mark"]


def _replace_second(template: str, marker: str, new_marker: str) -> str:
    """Replace the second occurrence of `marker` in `template` with `new_marker`."""
    first = template.index(marker)
    rest = template[first + len(marker):]
    second_in_rest = rest.index(marker)
    return (template[: first + len(marker)]
            + rest[:second_in_rest]
            + new_marker
            + rest[second_in_rest + len(marker):])


def build_prompt_pairs(n: int, seed: int = SEED):
    """Build n records {clean, corrupted, IO, S}.

    Clean      = IOI prompt. The placeholder that appears twice is the subject S
                 (repeated name). The placeholder appearing once is IO = correct answer.
    Corrupted  = same template but the second occurrence of the repeated placeholder
                 is replaced by a third name C — no name repeats, IOI signal destroyed.

    ABBA templates have {B} twice → IO=A, S=B.
    BABA templates have {A} twice → IO=B, S=A.
    """
    random.seed(seed)
    templates = TEMPLATES_ABBA + TEMPLATES_BABA
    seen = set()
    pairs = []
    while len(pairs) < n:
        template = random.choice(templates)
        a, b, c = random.sample(NAMES, 3)

        if template.count("{B}") == 2:
            io, s = a, b
            corrupted_template = _replace_second(template, "{B}", "{C}")
        elif template.count("{A}") == 2:
            io, s = b, a
            corrupted_template = _replace_second(template, "{A}", "{C}")
        else:
            raise ValueError(f"unexpected template (no repeated placeholder): {template!r}")

        clean = template.format(A=a, B=b)
        corrupted = corrupted_template.format(A=a, B=b, C=c)

        key = (clean, corrupted)
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"clean": clean, "corrupted": corrupted, "IO": io, "S": s})
    return pairs


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def logit_diff(logits: torch.Tensor, io_tok: torch.Tensor, s_tok: torch.Tensor) -> torch.Tensor:
    """Per-prompt logit diff at the final position. Returns shape (batch,)."""
    last = logits[:, -1, :]                       # (batch, vocab)
    io = last.gather(-1, io_tok[:, None]).squeeze(-1)
    s = last.gather(-1, s_tok[:, None]).squeeze(-1)
    return io - s


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def hook_zero_head(z, hook, head_idx):
    z[:, :, head_idx, :] = 0.0
    return z


def hook_mean_head(z, hook, head_idx, mean_z):
    # mean_z shape (seq_len, d_head) — broadcast across batch
    z[:, :, head_idx, :] = mean_z.to(z.dtype).to(z.device)
    return z


def hook_patch_head(z, hook, head_idx, corrupted_z):
    # corrupted_z shape matches z[:, :, head_idx, :] = (batch, seq, d_head)
    z[:, :, head_idx, :] = corrupted_z.to(z.dtype).to(z.device)
    return z


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print(f"device: {DEVICE}")
    torch.set_grad_enabled(False)

    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    pairs = build_prompt_pairs(N_PROMPTS)
    print(f"Built {len(pairs)} prompt pairs.")
    for p in pairs[:3]:
        print("  clean    :", p["clean"])
        print("  corrupted:", p["corrupted"])
        print("  IO/S     :", p["IO"], "/", p["S"])
        print()

    # Tokenize clean / corrupted, enforce equal lengths within each group
    clean_tokens = model.to_tokens([p["clean"] for p in pairs])
    corr_tokens = model.to_tokens([p["corrupted"] for p in pairs])
    assert clean_tokens.shape == corr_tokens.shape, (
        f"shape mismatch: clean={clean_tokens.shape} corr={corr_tokens.shape}")
    print(f"clean_tokens: {tuple(clean_tokens.shape)}")

    io_tok = torch.tensor(
        [model.to_single_token(" " + p["IO"]) for p in pairs], device=DEVICE)
    s_tok = torch.tensor(
        [model.to_single_token(" " + p["S"]) for p in pairs], device=DEVICE)

    # Baseline on clean prompts
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    baseline_ld = logit_diff(clean_logits, io_tok, s_tok)
    baseline_mean = baseline_ld.mean().item()
    baseline_std = baseline_ld.std().item()
    print(f"\nBaseline clean logit_diff: {baseline_mean:.3f} ± {baseline_std:.3f}")

    # Corrupted baseline (for reference)
    corr_logits, corr_cache = model.run_with_cache(corr_tokens)
    corr_ld = logit_diff(corr_logits, io_tok, s_tok)
    print(f"Corrupted  logit_diff:    {corr_ld.mean().item():.3f} ± {corr_ld.std().item():.3f}")

    # Pre-compute means (over batch, per (seq_pos, d_head)) for mean ablation
    # cache[("z", layer)] has shape (batch, seq, n_heads, d_head)
    mean_z_by_layer = {}
    for layer in {l for l, _ in ALL_HEADS}:
        z_clean = clean_cache["z", layer]              # (batch, seq, H, d)
        mean_z_by_layer[layer] = z_clean.mean(dim=0)   # (seq, H, d)

    # Pre-extract corrupted z per layer
    corr_z_by_layer = {layer: corr_cache["z", layer] for layer in {l for l, _ in ALL_HEADS}}

    results = {"baseline_clean": baseline_mean,
               "baseline_clean_std": baseline_std,
               "baseline_corrupted": float(corr_ld.mean().item()),
               "heads": {}}

    for (layer, head) in ALL_HEADS:
        key = f"L{layer}H{head}"
        head_result = {}

        # Zero ablation
        out = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(tl_utils.get_act_name("z", layer),
                        partial(hook_zero_head, head_idx=head))],
        )
        ld_zero = logit_diff(out, io_tok, s_tok)
        head_result["zero"] = {
            "mean": float(ld_zero.mean().item()),
            "std": float(ld_zero.std().item()),
            "delta": float(ld_zero.mean().item() - baseline_mean),
            "pct_recovered": float((ld_zero.mean().item() - corr_ld.mean().item())
                                    / (baseline_mean - corr_ld.mean().item())) * 100.0,
        }

        # Mean ablation
        out = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(tl_utils.get_act_name("z", layer),
                        partial(hook_mean_head, head_idx=head,
                                mean_z=mean_z_by_layer[layer][:, head, :]))],
        )
        ld_mean = logit_diff(out, io_tok, s_tok)
        head_result["mean"] = {
            "mean": float(ld_mean.mean().item()),
            "std": float(ld_mean.std().item()),
            "delta": float(ld_mean.mean().item() - baseline_mean),
            "pct_recovered": float((ld_mean.mean().item() - corr_ld.mean().item())
                                    / (baseline_mean - corr_ld.mean().item())) * 100.0,
        }

        # Activation patching (noising: patch clean run with corrupted z)
        out = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(tl_utils.get_act_name("z", layer),
                        partial(hook_patch_head, head_idx=head,
                                corrupted_z=corr_z_by_layer[layer][:, :, head, :]))],
        )
        ld_patch = logit_diff(out, io_tok, s_tok)
        head_result["patch_from_corrupted"] = {
            "mean": float(ld_patch.mean().item()),
            "std": float(ld_patch.std().item()),
            "delta": float(ld_patch.mean().item() - baseline_mean),
            "pct_recovered": float((ld_patch.mean().item() - corr_ld.mean().item())
                                    / (baseline_mean - corr_ld.mean().item())) * 100.0,
        }

        results["heads"][key] = head_result

        print(f"\n{key:7s}  baseline={baseline_mean:+.3f}  "
              f"corr={corr_ld.mean().item():+.3f}")
        print(f"          zero  : LD={ld_zero.mean().item():+.3f}  "
              f"Δ={head_result['zero']['delta']:+.3f}  "
              f"%recov={head_result['zero']['pct_recovered']:5.1f}")
        print(f"          mean  : LD={ld_mean.mean().item():+.3f}  "
              f"Δ={head_result['mean']['delta']:+.3f}  "
              f"%recov={head_result['mean']['pct_recovered']:5.1f}")
        print(f"          patch : LD={ld_patch.mean().item():+.3f}  "
              f"Δ={head_result['patch_from_corrupted']['delta']:+.3f}  "
              f"%recov={head_result['patch_from_corrupted']['pct_recovered']:5.1f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
