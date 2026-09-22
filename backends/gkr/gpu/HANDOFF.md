# a-gpu handoff (2026-09-22 ~08:00Z)

State: **done for the overnight goal.**  Candidate A's prover runs on the H100 for the full K=1536 VU, B = 4096,
over BabyBear^6, verified by the Python verifier, 4.43 s median (v2e), 52/52 negatives rejected, six ledger entries
(`backends/numerical/reports/ledger/a-gpu.jsonl`, run ids r20260922-065037-41f8, -070013-b446, -071327-e0d0,
-072549-570f, -073524-64ff, -074455-d0b6).  Branch `lane/a-gpu`, all work committed; nothing is running on `vy-g5`.

## Where things are

* Pod `vy-g5` (`/tmp/g5.sh` = ssh helper on the laptop): `/workspace/verity/backends/gkr/gpu` is a tar-synced copy of
  the worktree (`/tmp/g5sync.sh`), `/workspace/venv312/bin/python` (torch cu124, Triton 3.x, cupy-cuda12x),
  `/workspace/bb/pos4096` (4096 exported `vu-k1536` VUs, BabyBear embedding), `/workspace/bb/neg` (52 negatives),
  `/workspace/bb/run4096_v*.json` (per-version records), `/workspace/bb/neg_v2d.json` (52/52 rejected on the kernel
  path).  Recorded runs go through `research run --on vy-g5 --project verity --source . --env PYTHONPATH=backends/gkr
  --stage gkr.gpu.vX -- bash -c '/workspace/venv312/bin/python -m gpu.run prove /workspace/bb/pos4096 --device cuda
  --reps 3 --json $RESEARCH_RUN_DIR/run.json'` (commit first; `research fetch <id> --all` to pull the JSON).
* Triton kernels compile on first use (the R16 LogUp kernels take ~3 min the first time; cached in `~/.triton`).
* Profiling helpers used tonight (not in the repo): `/workspace/prof.py <dir>` wraps the prover's functions with
  synced timers; `/workspace/prof3.py <dir> <prover function>` runs `torch.profiler` on one function.

## Next three steps

1. **Round overhead**: the prover is now bound by 1920 sequential transcript rounds (each: kernel + partial-sum
   reduction + D2H sync + Python transcript) and ~10k small launches.  Fuse the fold into the round kernel (the
   round reads the tables anyway; the fold for the *next* round can be written by the *same* launch once the
   challenge is known, i.e. restructure as: round kernel writes partials; host derives `r`; one launch folds all
   five LogUp tables / both GKR tables + eq), then CUDA-graph the per-round sequence.  Expected: lookup 1.4 -> ~0.5 s,
   arith 1.07 -> ~0.5 s.
2. **Opening + commit**: `ligero.open_w_qc` is 0.84 s of fp64 tensor-core matmuls at the fp64 peak; move `mm_mod` to
   INT8 tensor cores (`torch._int_mm` with 8-bit limbs and offset correction, 16 GEMMs at ~30x the fp64 rate) or
   to a batched NTT; the commitment's torch radix-2 NTT (0.38 s) becomes one INT8 matmul too (`X @ NTT matrix`).
   Merkle paths + column extraction on the host (~0.4 s) should stay on the device (`hash_gpu` levels indexed by
   tensor).
3. **Verifier**: 7 s in Python (transcript replay, `hashlib` Merkle paths, 33.9 MB of proof as Python tuples); port
   to Rust over BabyBear^6 (the crate's `main.rs` structure is already mirrored 1:1 by `prover.verify`) or at
   least vectorise the eq-point / interpolation loops.  Also apply the sub-16-bit range-check fix (DESIGN §1) to
   the Goldilocks `gkr_export.CircuitCtx` on the CPU lane.
