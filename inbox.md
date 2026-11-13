[2026-11-13T01:35Z] af2da817227d
TOPCODE, Thu 5:35 PM PST

**The guard is in its own commit (45c881f613), and the route, its tests and the 81M entry are pushed on `cursor/training-topcode-9478`; my entry's comparison is running, and the every-entry comparison launches when it shows the memory to ask.**

Commits on `cursor/training-topcode-9478`, its base GROW's b16502a23d (unchanged since):
- d64f33f928, then 574bab9f04 (a lint rename only): the `norm` rule, beside the record's plain normalisations, with its test. The present tests pass unchanged. RAN: `tests/test_of_record.py` on my VM, 19 pass.
- 45c881f613: the guard `(cfg[k][0] - n) % excess[scale][0] == 0` over the held backward keys, with the detached case as its test. RAN: without the guard, that test fires the rule; with it, the rule doesn't fire. The real entry's counts come from the comparison below; if they break the guard, I'll send you the counts and leave the guard as it is.
- dab4575423: `routes/topcode.py` (route `VERSION` 1, package still 4.24.0), with the packing in `codecs.wire` (`pack_codes`, `unpack_codes`, which refuse a width not a multiple of 8). R is a registered buffer built once in `build`, seeded 1000 + grown depth. The head is `model.top(h, len(model.blocks))`. The backward rounds on its own grid of 2^bits − 1 steps, not through bytes. Its tests, among them: the logits equal θ's bit for bit at the first step (f32 and bf16); R·Rᵀ = I within 1e-6, and the same R at every build; the round trip at 4 to 8 bits; and the tiny trace with no `refused`, `data_dependent` or `carried_scalars`, at `a_bits` = d·bits + 16.
- 14cf2170a1: the 81M entry's reference in `test_entries.py` and its entry in `of_record.toml`, with counted = 3 for now and no `record_only` yet.

RAN r20261113-013251-e432 (smoke run): a pool job takes only `--cwd source`. Torch 2.14.0+cpu loads on the node, and the record fetches with sha256 9dd58783…, as the brief says.

RAN r20261113-013457-5fe2 (running): of_record on my entry at counted 3, and at counted 4 from a scratch copy of the data file, each with `/usr/bin/time -v`. It gives condition 9's evidence of which first steps differ, the `record_only` lines, the guard on the real counts, `a_bits` against 3,088, and the memory for the every-entry jobs.

A finding outside my change. RAN on my VM: on torch 2.14.0+cpu at 4 threads, bf16 SDPA sometimes returns a different value for one tile of the same input (about once in 400 calls); at 1 thread it is deterministic. The suite's conftest pins 1 thread only in pytest's own process, so a job `run.main` starts in a child process uses every core. My slow resume test with its trace on diverged at step 2 for this reason, so it sets `OMP_NUM_THREADS=1` and its docstring says why; it then passed 3 times out of 3. INFERRED: the existing slow resume tests that start child processes carry the same risk. I haven't changed them.

