#!/bin/bash
# push-site.sh — publish site content to origin/main without notebooks
set -e

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"

echo "==> Syncing site to origin/main..."

# 1. Merge backup into main (no-commit so we can clean up)
git checkout main
git merge backup --no-commit --no-ff 2>/dev/null || true

# 2. Restore main's .gitignore (with notebooks/ line, not backup's version)
git checkout HEAD -- .gitignore

# 3. Remove any notebooks files that got staged from the merge
git rm -r --cached notebooks/ 2>/dev/null || true

# 4. Commit and push if anything changed
if ! git diff --cached --quiet; then
    git commit -m "sync from backup $(date +%F)"
    git push origin main
    echo "==> Pushed to origin/main."
else
    echo "==> Nothing new to publish."
fi

# 5. Return to backup
git checkout backup
echo "==> Back on backup."