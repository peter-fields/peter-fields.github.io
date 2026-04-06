---
name: memory mirror sync
description: How memory file syncing to notebooks/memory_mirror/ works — do NOT run cp manually
type: feedback
---

Do NOT run `cp` manually to sync memory files to `notebooks/memory_mirror/`. A PostToolUse hook in `~/.claude/settings.json` auto-copies any edited memory file to the mirror whenever Edit or Write is used on a file in the memory directory.

**Why:** Running cp manually triggers an unnecessary permission prompt and is redundant with the hook.

**How to apply:** After editing any memory file with Edit or Write, the sync is already done. No follow-up cp needed.
