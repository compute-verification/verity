[2026-11-13T03:28Z] 2e0691180122
From top, Thu 12 Nov 7:29 PM PST: #9375 (the remote merge) reached main at 7:01 PM PST (66dce017d86 + lock records 06365af5a3b8); main is at 1e4d5c993c0 or later. Build and certify the chain on main's tip as it is now, not on a base before 06365af5a3b8: regenerate the generated commits there, and in tests/test_boundaries.py PLACE keep both lines if it stops. Name the base SHA in the PR body. Nothing else in your brief changes.

