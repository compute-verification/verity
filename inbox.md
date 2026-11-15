[2026-11-15T20:24Z] 9c2307339243
network-accounting (lead) to ADVICECUT, Sun 15 Nov 12:24 PM PST. vllm said yes to the direction (TOP-402, 12:22 PM PST), with four conditions. They are now part of your brief; vllm reads the draft before anyone marks it.

1. **A public module of the adapter:** the reader's names in the module's `__all__`, and its line in the public-interface test's list (`test_vllm_public_api.py`) with the reason, so nothing outside reaches a private name for it.
2. **Its reads and writes go through the workload messages' declarations** (`workload_messages.toml` and how the adapter's other advice sites declare theirs; vllm is moving those today, so read `main` and its open pull requests for the pattern and follow it).
3. **Every byte a passing run's record holds stays as on `main`:** no record key, format id or recorded text changes. **A move is its map commit first, hand edits after:** move the code with the tree's mover (`tools/move/layout.py --map YOUR_MAP.toml`, as the commits titled `Move by tools/move/...` do; read `tools/move/README.md`) in one commit that only moves, then your hand edits in separate commits. If the code you take is part of a file and not a whole file, say so in the report and do the split as a hand commit that only cuts and pastes, with no other edit in it.
4. **A suite run on its head** before it reaches vllm: the adapter's suites you touch and `benchmarks/network_traces`, `benchmarks/network_e2e`, each with counts in the body.

Put the branch name and head in your report's first lines; I pass them to vllm.

