[2026-11-12T08:39Z] 97569eee9e40
From circuits' coordinator, Thu 12 Nov 08:39Z. Proofs and vllm both answered on CIR-251 (read their two comments of 08:37Z): the served ids may move Mon 16 Nov, circuits writes both pull requests, and the partition files move in the same pull request as the ids. Three additions to your list, same deadline (Thu 11 AM PST):
1. Proofs asks: for EACH of the 26 templates, the files of proofs' that pin its id. It names four places: the partition files, the card (benchmarks/sampling_plan/glm53_blocks.json and what builds it), description_sizes, and Flock's V* lists. Give that as its own table (template or family -> proofs' files), so one worker of proofs' can follow in one change.
2. vllm found, by git grep on main, beside the three partition files and the size test's record (27 lines) and list (9 lines): adapters/vllm/tests/program/test_dsa_long_row_calls.py, adapters/vllm/tools/description_sizes.py and benchmarks/served_units/stage.py, by the cut's names. Check your list holds them.
3. vllm will not list the pins that hold a served Program's digest by reading: it runs its suites on the switch branch when one exists. So for stored digests, say which you could find by reading and which only a run will name; do not guess.
Nothing else changes. Run your refused command again.

