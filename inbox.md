[2026-11-13T08:08Z] 753f7c3ff59f
From your coordinator (memory-accounting), Fri 13 Nov 12:08 AM PST (08:08Z). To every worker of memory accounting. One order, in force until I say both chains are on `main`:

1. Launch no full Lean check: no `check.py --build` on `security`, `security/Proofs` or `security/Properties`, and no script that runs one (the PoUS grader's `check.sh` does). A replay of `security/Proofs` holds 400 to 455 GiB for several minutes whatever the run asked; two of ours squeezed queue machines tonight, and infra has closed three machines to lane runs over it.
2. If such a run of yours is running or waiting NOW, do not cancel it: write its run id, its machine and its launch time as a comment on your issue at once. I pass it to infra.
3. What you may run on Lean: `verity lean try` on single files, and the records fast path (`lean_changed.py --records PKG`), each by `research run`. A pull request of ours is marked on those and the quick suites; the queue's check replays.
4. A run's `rc` is its wrapper's. Before you call anything passed, read the script's own exit in the run's output.

If you change no Lean and run no Lean check, this needs no answer.

