# Dev Setup & Git Workflow

## Jekyll Setup (fixed Feb 2026)
- **Problem**: `github-pages` gem pins `jekyll-sass-converter` 1.x which uses deprecated `sass` gem (Ruby Sass). That gem can't handle UTF-8 characters (`\xE2`) in Minimal Mistakes' vendored Susy SCSS files. Build crashes with `Invalid US-ASCII character "\xE2"`.
- **Fix**: Replaced `github-pages` with standalone Jekyll 4 + `jekyll-sass-converter ~> 3.0` (uses `sass-embedded`/Dart Sass). Added `jekyll-seo-tag` and `jekyll-remote-theme` explicitly to Gemfile.
- **Sass deprecation warnings** from the theme are harmless noise, don't affect rendering.
- GitHub Pages deployment is unaffected — it uses its own build pipeline.

## Git Branches
- **Always work on `backup` branch** — notebooks are tracked here, `.gitignore` has no `notebooks/` line
- `main` branch — `notebooks/*/scratch/` and `notebooks/anthropic_app/` excluded (`.gitignore`); `notebooks/*/final/` IS tracked and public (for nbviewer)

## Remotes
- `origin` → public repo: `peter-fields/peter-fields.github.io` (GitHub Pages)
- `private` → private repo: `peter-fields/site-private`

## Daily Workflow
```bash
# Normal work: commit and push notebooks + site to private backup
git add <files> && git commit -m "..."
git push                        # goes to private/backup (tracking is set)

# Publish site to GitHub Pages (strips notebooks automatically)
./push-site.sh                  # merges backup→main, removes notebooks, pushes origin/main, returns to backup
```

## Git Gotchas
- `git push` from `backup` always goes to `private/backup` (upstream is set)
- `push-site.sh` is at repo root; handles `.gitignore` merge conflict automatically
- Never manually commit site content directly on `main` — always use `push-site.sh`
- `.vscode/` is tracked on both branches; `.DS_Store` is ignored everywhere (`**/.DS_Store` in both `.gitignore` files)
- **push-site.sh gotcha**: `git merge` won't auto-stage files that are new to `main` (never existed there before). `push-site.sh` has a step 3b that explicitly runs `git checkout backup -- notebooks/*/final/` to handle this. If a final notebook isn't showing up on main after push-site, run that command manually on main then commit+push.
- **nbviewer URL pattern**: `https://nbviewer.org/github/peter-fields/peter-fields.github.io/blob/main/notebooks/<post-slug>/final/<notebook>.ipynb`
