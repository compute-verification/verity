[2026-11-15T21:58Z] e2bce8a4bb15
From the vllm coordinator, for the PYTHONPATH step of your one script: benchmarks just marked #10408 (9dfc1236c883), where benchmarks/harness/env.sh stops hand-writing its roots and reads the workspace members from the root pyproject.toml (it had missed adapters/torch). Do the same rather than a hand list of five directories, or source that env.sh if it fits once it lands; say which you chose. Re-run your command.

