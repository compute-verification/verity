[2026-11-16T04:15Z] a00c4c5414b0
From the proofs coordinator, a tip from a colleague on the sibling branch: setup.sh restores the cache only when the branch contains a recent origin/main; from an older base a run compiled about 4,000 modules for over 30 minutes. If your first run goes cold, merge origin/main into your branch (nothing under security/Proofs changed on main beyond #10416, which is your base's own work) before 08:00Z, and not after that hour until I say: a tree-wide move lands about 09:00Z.

