#!/bin/bash
# push-site.sh — publish site content to origin/main without notebooks
set -e

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"

echo "==> Syncing site to origin/main..."

# 1. Merge backup into main (no-commit so we can clean up)
git checkout main
git merge backup --no-commit --no-ff 2>/dev/null || true

# 2. Resolve any merge conflicts: for files deleted in HEAD (main), keep them deleted
#    This handles rename/delete and modify/delete conflicts from notebook reorganization
CONFLICTED=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
if [ -n "$CONFLICTED" ]; then
    echo "$CONFLICTED" | xargs git rm --cached -f 2>/dev/null || true
    # Also remove any conflict-staged working tree files
    echo "$CONFLICTED" | while read f; do
        [ -e "$f" ] && git checkout HEAD -- "$f" 2>/dev/null || rm -f "$f"
    done
fi

# 3. Restore main's .gitignore
cat > .gitignore << 'EOF'
_site
.sass-cache
.jekyll-metadata
Gemfile.lock
**/.DS_Store
notebooks/*/scratch/
notebooks/anthropic_app/
notebooks/memory_mirror/
notebooks/other_jobs/
notebooks/phase_readout/
notebooks/post4_qk_metric/
notebooks/tensor_notation/
notebooks/interp_prep/
EOF
git add .gitignore

# 4. Remove all notebook files from staging except post2/final
git rm -r --cached notebooks/ 2>/dev/null || true

# 5. Explicitly stage the one allowed notebooks path
git checkout backup -- notebooks/post2_attention-diagnostics/final/ 2>/dev/null || true

# 6. Commit and push if anything changed
if ! git diff --cached --quiet; then
    git commit -m "sync from backup $(date +%F)"
    git push origin main
    echo "==> Pushed to origin/main."
else
    echo "==> Nothing new to publish."
fi

# 7. Return to backup
git checkout backup
echo "==> Back on backup."
