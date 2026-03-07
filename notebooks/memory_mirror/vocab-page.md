# Vocab Page Spec

## Concept
A curated vocabulary collection page — been maintaining for 2+ years. I am a "word collector": when I encounter a precise/expressive word while reading, I add it. Framing: **curious collector**, not "quirky word nerd." Good vocabulary = compression, not ornament.

## Intro text (approved)
> I keep a running list of words that feel unusually precise or conceptually useful. Whenever I encounter one while reading, I add it here. For me, good vocabulary is less about ornament and more about compression: the right word can make a pattern easier to notice or an idea easier to reason about.

## Data format — vocab.json
```json
{
  "word": "laconic",
  "definition": "...",
  "pos": "adjective",
  "example": "...",
  "added": "2026-02-20"
}
```

## Features (priority order)
1. **Word of the Day** — deterministic RNG seeded by date; shows word + definition + optional POS + example
2. **Random Word button** — pick random entry, display word/definition/example
3. **Searchable list** — filter by substring, click to expand definition; pure JS
4. **Word Cloud** — font sizes by RNG; mode TBD (random per load or deterministic per day)
5. **Recently Added** — sort by `added` date, show newest N entries

## Jekyll/implementation notes
- Lives at `/vocab/` — needs a `vocab.md` page with `layout: single` or custom layout
- Vocab data in `_data/vocab.json` (Jekyll can use `site.data.vocab`) or static JSON fetched by JS
- Word of the Day logic: deterministic hash of today's ISO date → index into sorted word list
- All interactivity is client-side JS (no backend)
- Add "Vocab" to top nav in `_config.yml` navigation

## Personal notes
- Know many words but not a great speller (vocab knowledge and orthographic recall = separate processes)
- Run spell check on vocab JSON before committing
- List is actively maintained — "recently added" section signals this

## Nav target structure
```
Home | Research | Publications | Projects | Vocab | Writing
```
