# Post Writing Workflow

## Front matter template
```yaml
---
title: "Post Title"
layout: single
author_profile: false
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
sidebar:
  - title: "Notation"
    text: |
      (notation relevant to the post)
tags: [tag1, tag2]
excerpt: "One-paragraph summary for the homepage listing."
---
```

## Key conventions
- **Excerpts**: Use the `excerpt:` front matter field to control what shows on the homepage. No character limit (we removed truncation in `_includes/archive-single.html`). Don't use `<!--more-->` — it's redundant.
- **MathJax**: Set `mathjax: true` in front matter. Use `$$...$$` for display math, `\\(...\\)` for inline math. Use `\label{eq:name}` and `\eqref{eq:name}` for equation cross-references.
- **Footnotes**: Use `[^N]` inline and `[^N]: text` at the bottom. Keep numbering sequential.
- **Sidebar notation**: Include a notation legend in the sidebar for math-heavy posts. Group by topic (Tokens & Embeddings, Attention Components, Distributions, Diagnostics, Other).
- **File naming**: `_posts/YYYY-MM-DD-slug.md` — the date in the filename is the publish date.
- **Layout**: `single` layout with `toc: true` and `toc_sticky: true` for long posts.

## Local preview
- `bundle exec jekyll serve` (or `jserve`) → http://127.0.0.1:4000
- Post edits auto-regenerate; `_config.yml` changes require restart
- `_includes/archive-single.html` overrides the theme to remove excerpt truncation
- If startup fails with "Address already in use": `lsof -ti :4000 | xargs kill -9` then retry
- `{% post_url %}` tags require the **exact filename date** (e.g. `2026-02-17-why-softmax`)
