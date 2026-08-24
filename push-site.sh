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
#    `git rm --cached notebooks/` (step 4) untracks those files on main but leaves
#    them on disk. Switching back to backup — where they ARE tracked — then aborts
#    with "untracked working tree files would be overwritten". Their on-disk content
#    is identical to backup's (main never modifies them), so -f is safe here.
#    Guard: only force if the sole differences are those notebooks paths.
if ! git checkout backup 2>/dev/null; then
    echo "==> Untracked notebooks files blocking checkout; verifying they match backup..."
    MISMATCH=0
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        if [ "$(git hash-object "$f")" != "$(git rev-parse "backup:$f" 2>/dev/null)" ]; then
            echo "    DIFFERS from backup: $f"
            MISMATCH=$((MISMATCH + 1))
        fi
    done < <(git ls-tree -r --name-only backup -- notebooks/)

    if [ "$MISMATCH" -eq 0 ]; then
        git checkout -f backup
        echo "==> Restored backup (forced; on-disk notebooks matched backup exactly)."
    else
        echo "==> ABORT: $MISMATCH notebooks file(s) on disk differ from backup."
        echo "    Still on main. Inspect the files above, then run: git checkout -f backup"
        exit 1
    fi
fi
echo "==> Back on backup."
