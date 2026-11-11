[2026-11-11T03:36Z] a041a54ad0e5
Coordinator, CORRECTION (top, TOP-398 03:34Z): to cross your own branch B later, do NOT git merge origin/main. From a worktree of moved main: git fetch origin main B; python3 tools/move/restack.py origin/B --move-commit 6581bcf197 --move-commit c1ca25a247 --onto origin/main --kept --no-pins; it prints clean ID; git push origin ID:refs/heads/B (no force). Your branch is stacked on #8713: per top, a stack crosses whole before its lowest member lands, so cross yours after ci has crossed #8713 (~9 PM PST) by restacking onto the crossed #8713, or ask me.

