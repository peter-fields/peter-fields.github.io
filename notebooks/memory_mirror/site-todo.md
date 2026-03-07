# Site Maintenance & Improvement TODO

## Priority 1 — Site credibility — DONE (2026-03-03)

### Identity & trust signals
- [x] Create **About** page — `_pages/about.md`, live at /about/
- [x] Google Scholar + GitHub links already in author sidebar (`_config.yml`)
- [x] Short site description already in `_config.yml`
- [x] Profile picture — `assets/images/bio-photo.jpg`, avatar uncommented in `_config.yml`

### Navigation cleanup
- [x] Top nav: Posts | About | Vocab (Tags removed from header)
- [ ] Verify TOC depth is reasonable (`max_level: 2` if crowded)
- [ ] Confirm sidebar notation renders cleanly on mobile

## Priority 2 — Technical robustness

### MathJax stability
- [ ] Spot-check equations on Chrome, Firefox, mobile
- [ ] Confirm all `\eqref` links work after full deploy
- [ ] Standardize math conventions (inline: `$…$`, numbered display: `$$ … \label{} $$`, multiline: `aligned` inside `$$`)

### Repository hygiene
- [ ] Confirm `.gitignore` excludes `.bundle/`, `_site/`, `vendor/bundle/`
- [ ] Keep committed: `.ruby-version`, `Gemfile`, `Gemfile.lock`
- [ ] Verify `bundle exec jekyll serve` works from clean terminal

## Priority 3 — Discoverability

### Metadata
- [ ] Add meta description to post front matter
- [ ] Consider adding `mechanistic-interpretability` tag
- [ ] Enable tags page (Minimal Mistakes supports this)

### Social preview (optional)
- [ ] Add default social image in `_config.yml`
- [ ] Verify OpenGraph preview looks reasonable when sharing

## Priority 4 — Infrastructure for future posts

### Series structure
- [ ] Decide on series tag (e.g. `categories: [softmax-lens]` or `series: softmax-lens`)
- [ ] Create consistent post naming pattern
- [ ] Document math style in local NOTES.md

### Authoring workflow sanity
- [ ] Confirm VS Code preview + site rendering both work
- [ ] Confirm new posts auto-appear in index
- [ ] Confirm TOC behaves well for long posts

## Priority 5 — Personality / Personal sections

### Vocab page (see [vocab-page.md](vocab-page.md) for full spec)
- [ ] Create `_data/vocab.json` with initial entries (word, definition, pos, example, added date)
- [ ] Create `vocab.md` page with layout + JS (word of the day, random word, search, word cloud, recently added)
- [ ] Add "Vocab" to top nav in `_config.yml`
- [ ] Spell-check vocab entries before committing

### Writing / Substack
- [ ] Add "Writing" nav item linking to Substack (once it exists)
- [ ] Substack topics: Christianity from a scientific perspective, philosophy of language, epistemology
- [ ] Tone: analytical, exploratory, non-combative; lean into epistemic friction framing
- [ ] Keep Substack link visible but secondary on site

---

## NOT worth time right now
- heavy CSS/theme customization
- analytics tooling
- comment systems
- SEO rabbit holes
- performance tuning

## Definition of done
- Someone lands on site and immediately sees who you are
- Equations render cleanly everywhere
- Navigation is obvious
- New posts are frictionless to publish
- Currently ~80-85% there
