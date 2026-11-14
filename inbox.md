[2026-11-14T16:56Z] 5ecbd3837c69
INF-861 UNCOVERED: from the lead (infra). One more thing for your PR, since you are in the same files: #10016 landed (16:36Z) and the queue's ratchet gate says it added 7 retired-term sites in `tools/research/tests/test_retention_unpreserved.py` (about lines 780 to 790: the word "cap", e.g. `cap = 5`), as `tests/lint/test_repo_retired_terms.py` counts them. Fix them forward in your branch (the Glossary's word is "limit"), run that lint test, and say so in the body. Merge `origin/main` first: #10016 is on it.

