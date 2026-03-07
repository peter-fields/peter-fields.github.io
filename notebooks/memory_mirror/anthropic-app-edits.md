# Anthropic App — Final Edit Checklist

Generated from final pass review (March 2026). All items complete.
Essays file: `notebooks/anthropic_app/essays.md`.

---

## ESSAYS

### Why Anthropic?
- [x] **1. "none may be approached"** → "can only be provisionally addressed until we understand how AI actually works"
- [x] **2. "Technological advancement..." sentence** → replaced with "Many technological transitions have moved faster than our ability to understand them; I don't think AI should."
- [x] **3. "at scale" ending** → cut; closing rewritten to "Anthropic is where I can apply these skills to larger models, harder questions — and with a team that understands the scale of collaboration required."
- [x] **4. Add big-science / collaboration signal** → handled in closing sentence above
- ~~**4b. "information geometry"**~~ — Peter's intentional; fine if he can speak to it in interview.

### Why interpretability team?
- [x] **5. "computationally expensive"** → "Verifying QK circuit membership requires causal interventions that don't scale to large models."
- [x] **6. Add Mathematical Framework cite** → added "(Elhage et al., 2021)" after "QK circuits"

### Work sample (link + description)
- [x] **7. "uniform distribution over query-key scores"** → "uniform distribution over token positions"
- [x] **8. "stability"** → "susceptibility"
- [x] **9. "This blogpost utilizes"** → "This post applies"
- [x] **10. "the missing attention gap currently unaddressed by CLTs"** → "the QK circuit gap currently unaddressed by CLTs"
- [x] **11. Add Python/TransformerLens mention** → "implemented in Python using TransformerLens" added

### Technical achievement
- [x] **12. "brought this heuristic into the light..."** → "gave this heuristic a principled explanation"
- [x] **13. Significance ending** → "allows practitioners of temperature tuning to understand why it is required in the first place and what underlying bias it corrects"

### Additional information
- [x] **14. "philosophy of language" line** → CUT entirely
- [x] **15. Add honest-null signal** → SKIPPED (Peter's call — felt unnecessary)

---

## RESUME
- [x] **16. Python/TransformerLens** → added to Summary bullet ("implemented in Python using TransformerLens") and Skills ("Python (PyTorch, TransformerLens)")

---

## POSTS

### Post 2 (`_posts/2026-02-24-attention-diagnostics.md`)
- [x] **17. χ vs ∂ρ̂ notation bridge in Recap** → added "Note: Post 1 used the notation ∂ρ̂ for the temperature susceptibility; here we write χ = ∂ρ̂/(log n)²."

### Post 1 (`_posts/2026-02-17-why-softmax.md`)
- [x] **18. "tokens (e.g. words from an LLM)"** → cut the parenthetical

---

## NEXT STEP — Final Coherence Check (new conversation)

Start a fresh conversation and ask it to do a final pass over the full application checking for:
- **Coherence**: do essays, posts, and resume tell a consistent story?
- **Punchiness**: are sentences doing real work or padding?
- **Personality**: does it sound like Peter or a generic applicant?
- **Not ChatGPT-ish**: flag over-polished, hedged, or AI-generated tone
- **Clarity**: is technical content understandable to an interpretability researcher?
- **Naivety**: any claims that would make an expert wince?

Files to review (same as MEMORY.md "Anthropic App — Final Pass" section):
- `notebooks/anthropic_app/essays.md` (v3 answers are live; v2 in HTML comments)
- `notebooks/anthropic_app/Fields_Peter_Resume_anthropic_final.pdf`
- `_posts/2026-02-17-why-softmax.md`
- `_posts/2026-02-24-attention-diagnostics.md`
- Notebook: `notebooks/post2_attention-diagnostics/final/attention_diagnostics_peter.ipynb`

All 18 edit items from this checklist are complete. The next pass is a fresh read for tone/coherence, not a bug-hunt.
