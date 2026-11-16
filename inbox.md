[2026-11-16T04:44Z] f4de50c9cdfe
From the vllm coordinator: a rule from top for tonights language pull requests (TOP-402, 8:43 PM PST): leave the lints list alone: NO change to tests/lint/allowlists/repo_retired_terms.json, and no --record or --prune. If your patch or apply_language.py touches that file or the lints list, take that out; say in your next report that it does not. Re-run your command.

