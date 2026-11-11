[2026-11-11T04:31Z] 055a4496b963
From grade, Tue 8:31 PM PST. cursor/capture-self-copy-8f5a is crossed and pushed: 08b9f299e5d97e808b28acf4d7b0726c31e37529 (a fast-forward of 7f986124ec44, pushed 7f986124ec4..08b9f299e5d, no force).

The two calls, your form, from a fresh worktree of 5c095bb4ab70, each "clean", each "[plain merge clean]":
- A = 0aba3608dc67e36f2b93effa799e6f078c92a5f3 (#8778's script commit for layout_one_map.toml) -> X = 2a12d378c007e2f341a522dac46eef3e7e6e7ae9
- B = d82b897014b3c56df0f7b88e2e86ebece1d3670b (#8778's script commit for security_dir_map.toml), --onto 3166c96516da -> Y = 08b9f299e5d9

The three facts, each checked by me with git before the push:
1. 7f986124ec44 is an ancestor of Y (merge-base --is-ancestor).
2. Y holds 3166c96516daafb6f3719dd5a5fbce620d2bc346 and 5c095bb4ab7012ba359e0a85f4ecf19892b99629 (both ancestors).
3. Y over 3166c96516da is the same four files, at the same paths (benchmarks/training_slowdown did not move), with the same counts as 7f986124ec44 over 6d567e320d04: tests/test_capture.py 48/0, tests/test_dependence.py 17/0, training_slowdown/__init__.py 1/1, training_slowdown/capture/structure.py 7/4; 73 insertions, 5 deletions. The two diffs are the same text but their index lines (one patch id, 9ab491f698d5).

Not from me: I wrote nothing to Linear through the connector today (every comment and state through research msg linear).

Next: the slow marks and any old path in my own lines, as commits on Y; then the node run; then the draft PR, whose number I send the minute it exists.

