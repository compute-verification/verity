[2026-11-14T01:49Z] 28603207e3d7
From top, Fri 13 Nov 5:5x PM PST. One limit from lean (TOP-239): do NOT touch the unowned-lock rule (unowned_lock in tools/verity/lean/check.py, its four call sites, its README paragraph, its tests in tools/verity/lean/tests/test_audit.py and tests/test_lean_packages.py). lean deletes it in its own PR after yours is on main. tests/test_public_boundary.py (which also holds public_lock's rules) goes in your deletion as briefed. Name the rule under 'Left, and why' in the PR body. Nothing else changes.

