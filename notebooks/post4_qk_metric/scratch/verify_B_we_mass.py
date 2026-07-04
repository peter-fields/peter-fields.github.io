"""Verification: W_E projection mass of B's top singular modes, BOTH models.

Reproduces the exp4 computation (B singular vectors vs the 90%-variance W_E
subspace) to settle which model the 'B routing modes live outside the embedding
space' result actually came from. Uses the March-saved B_crude_*.npy so the
numbers match what was originally reported.
"""
import os
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from transformer_lens import HookedTransformer
torch.set_grad_enabled(False)


def we_subspace(W_E, var_threshold=0.90):
    _, sv, Vt = np.linalg.svd(W_E, full_matrices=False)
    cumvar = np.cumsum(sv**2) / (sv**2).sum()
    k = int(np.searchsorted(cumvar, var_threshold)) + 1
    return Vt[:k].T, k


def we_projection_mass(vecs, V_we):
    proj = vecs.T @ V_we
    return (proj**2).sum(axis=1)


for model_name, bfile in [("gpt2-small", "B_crude_gpt2.npy"),
                          ("attn-only-2l", "B_crude_2l.npy")]:
    print("=" * 64)
    print(model_name)
    print("=" * 64)
    try:
        model = HookedTransformer.from_pretrained(model_name)
    except Exception as e:
        print(f"  could not load model: {type(e).__name__}: {e}\n")
        continue
    W_E = model.W_E.cpu().numpy()
    model = None
    print(f"  W_E shape: {W_E.shape}")

    B = np.load(bfile)
    print(f"  B_crude shape: {B.shape}")
    U_B, sv_B, Vt_B = np.linalg.svd(B)

    V_we, k = we_subspace(W_E, 0.90)
    print(f"  W_E 90%-var subspace dim: {k}")

    mass_u = we_projection_mass(U_B, V_we)   # query directions
    mass_v = we_projection_mass(Vt_B.T, V_we)  # key directions

    print(f"  B top-10 left-sv (query)  W_E mass: {mass_u[:10].round(4)}")
    print(f"  B top-10 right-sv (key)   W_E mass: {mass_v[:10].round(4)}")
    print(f"  B all-mode mean W_E mass: u={mass_u.mean():.4f}  v={mass_v.mean():.4f}")
    print()
