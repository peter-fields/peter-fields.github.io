---
name: temp-tune-publish-workflow
description: "How Peter publishes to Stephanie's public temp-tune repo (the PRR paper's code) — squash-merge PRs from temp branches into main; dev is the granular working branch."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 24e0f58e-a194-4c1e-a0de-757c991eef02
---

**Publishing to the PRR paper's public code repo — `origin` = github.com/sepalmer/temp-tune.**

Peter's established workflow (reconstructed from git history 2026-07-01 because he'd forgotten it): the public **`main`** branch gets **ONE squashed commit per change**, so it stays clean — "no sausage-making history." Evidence: every commit on `main` is a squash-merged PR — `finish ising code (#5)`, `finish toy model code (#3)`, `project setup (#1)` — with **zero merge commits** (squash, not regular merge).

- **`dev`** = granular working branch (many small commits; NOT published directly).
- **`main`** = public/default branch readers clone; the paper appendix links here. Receives only squashed commits.
- **To publish (his PR route):** branch a temp/"milestone" branch off `dev` (past names: `milestone-ising`, `milestone-toy-model`, `milestone-setup`), `git push origin <branch>`, open a PR into `main` on GitHub, and **Squash and merge** (yields `title (#N)`). Then delete the temp branch and resync local main (`git checkout main && git pull origin main`). Note: `gh` CLI is NOT installed → the merge click is done in the browser.
- **Equivalent local shortcut (no browser):** `git checkout main && git merge --squash dev && git commit -m "..." && git push origin main && git checkout dev`. Same single clean commit on main, but no PR-number trail.
- **`personal`** remote (peter-fields/temp-tune) = his private mirror ONLY; pushing there does NOT count as "published" — only `origin`/sepalmer does (the repo the appendix links to). See [[prr-paper-revision]].

⚠️ **Do NOT fast-forward or direct-push `dev` → `main`** — that dumps all the granular dev commits onto public main, which is exactly what the squash workflow exists to avoid.

## Remote topology (set 2026-07-01)
Three remotes, deliberate public/private split so both public faces stay clean and only `main` is exposed:
- **`origin`** = `github.com/sepalmer/temp-tune` — 🌐 PUBLIC, **`main` only** (default branch; the reproducibility artifact the paper appendix links to).
- **`personal`** = `github.com/peter-fields/temp-tune` — 🌐 PUBLIC, **`main` only** (Peter's **résumé/portfolio showcase** — must stay public + clean; do NOT make private).
- **`devbackup`** = `github.com/peter-fields/temp-tune-dev` — 🔒 **PRIVATE**, holds **`dev`** (granular WIP history). `dev` is set to track `devbackup/dev`.

**Rules:**
- Push `dev` **only to `devbackup`** (private). NEVER push `dev` to `origin` or `personal` (both public) — that's what exposes the sausage. (`dev` was deleted from both public remotes on 2026-07-01.)
- Publish to public `main` via the squash-PR workflow above; keep `personal`'s `main` in sync with `origin`'s (`git push personal main`) so the portfolio mirror shows the latest clean state.
- GitHub visibility is **per-repo, not per-branch** — this three-repo split is the only way to get "public main, private dev."
