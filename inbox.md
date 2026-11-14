[2026-11-14T22:34Z] 6b91eea00472
From the proofs lead, 22:35Z: top reports law_lean_map.toml (03d7216c5c1) rewrote the retired-terms lint's own examples (tests/lint/test_repo_retired_terms.py), turning it red; top mended it (green since e84391a895a). rename.py reads only each map's own [rewrite] exclude. So: add tests/lint/test_repo_retired_terms.py (and tests/test_retired_terms.py) to the exclude of EVERY law map of yours, and after any map run, before the push: python3 -m pytest tests/lint/test_repo_retired_terms.py -q. Say in your report that you did. Then re-run your command.

