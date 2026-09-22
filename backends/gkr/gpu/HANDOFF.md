# a-gpu-v2 handoff (2026-09-22 ~10:00Z) — checker v2 in the GPU prover

State: **done for the lane's goal, honest label.**  The checker-v2 circuit (PROTOCOL §14) runs end to end in this
prover over BabyBear^6 on the A100: full VU ×4096 in **5.73 s median of 3** (1.40 ms/VU, `overhead.vs_native_peak`
1.42e8) with the BabyBear-sound limb-form epilogue, `NON_ZK_PROOF_DIAGNOSTIC`; 4096/4096 accepted; 52/52
`vu-k1536-neg`, a-v2's 30 unit + 4 chain, and 15/15 epilogue-column negatives rejected.  Same-box v1 control 7.93 s
(1.38x).  Branch `lane/a-gpu-v2`, everything committed; nothing running on `vy-a100b`.  Ledger
`backends/numerical/reports/ledger/a-gpu-v2.jsonl` (5 entries; run ids r20260922-090930-3bc5 word-form v2,
-091544-4e02 v1 control, -095013-db80 limb-form v2; negatives -090151-07aa).  DESIGN §5b, README "Checker v2",
PROTOCOL §14.6 last bullet.

## What changed (all additive; the v1 path is byte-identical — the v1 control run proves it)

* `circuit.py`: `itable` lines (implicit tables) + `Table.materialise` on device (`shift`/`tnorm`, mirrors
  `circuit.rs::materialise`); `R<w>` range tables are created from the queries (`range_table_width`); use
  **`Table.nrows` / `Table.device_rows(dev)` / `Table.dense`**, never `len(t.rows)` (`rows` is `None` for implicit
  tables).  a-gpu2's `prove()` hunk has one `len(t.rows) % 2` — change it to `t.nrows`.
* `prover.py`: `Chain.links: list[Link]` (v2, `main.rs::add_chain` order — the 12-column strided read; `links is
  None` = v1 limb chain), `union_tables(segs)` (the epilogue's `R8/R15/R16` are not in the unit's list; the Rust
  a-v2 prover misses them — reported in DESIGN §5b), `spec_hash` absorbed at transcript start.
* `logup.py`: `table_rows` via `device_rows`; `multiplicities` by `bincount` for dense tables.
* `run.py`: v2 manifests (`variant: v2`, `chain_file`), `read_chain`, `negatives` over named dirs,
  `unit-negatives`, `epilogue-negatives`.
* `gpu/v2/`: `export.py` (BabyBear `FieldApi` for `gkr_export.v2`; `export|synth|negatives|stats`;
  `BabyBearV2Api("limbs"|"word")`), `epilogue_limbs.py` (b-air-v2's `pack_limbs`/`epilogue_limbs` verbatim +
  circuit/witness builders), `tests/`, `results/` (JSONs of every run, `z3_babybear_v2.json`).

## Pod layout (`vy-a100b`, A100-SXM4-80GB, `/workspace/venv312/bin/python`)

`/workspace/v2l/pos4096` (4096 `vu-k1536` VUs, limb epilogue) + `/workspace/v2l/neg` (52); `/workspace/v2/...` the
word-form set + `av2neg` (a-v2's generators); `/workspace/bb/pos4096` the v1 set; `/workspace/bench/v1` the built
bench-instances (the operand arrays are built, not committed).  Recorded runs: commit, then
`uv run research run --on vy-a100b --exclusive --project verity --source . --stage gkr.gpu.v2.X -- bash -c 'export
PYTHONPATH=$PWD/backends/gkr:$PWD/backends/numerical/python:$PWD/packages/verity/src; cd backends/gkr;
/workspace/venv312/bin/python -m gpu.run prove /workspace/v2l/pos4096 --device cuda --reps 3 --json
$RESEARCH_RUN_DIR/run.json'` (`gpu.run` alone needs only `backends/gkr`; the exporter needs all three).  Export:
`python -m gpu.v2.export export --root /workspace/bench/v1 --tier vu-k1536 --lo 0 --hi 4096 --out DIR --procs 48`
(~3 min).

## For a-gpu2 / a-fusion to absorb (the merge is mechanical: `git merge-tree` against both branches conflicts
only in the regenerated plot PNGs)

1. v2 per-bucket on the A100 at B=4096: lookup 2.85 / arith 0.36 / open 2.05 / commit 0.43 s.  **Lookup is the
   whole story**: 13 shallow trees (T_OP and SHIFT n=24, the rest 13–23), 3524 transcript slots (v1 1920), each
   round a kernel + reduction + D2H sync + Python transcript — round-bound, not bandwidth-bound (v1's lookup bucket
   is 2.22 s on the same box for 2x the leaves).  a-gpu2's one-launch-per-round + CUDA graphs and a-fusion's
   `GraphLogUp` apply directly; v2's implicit tables are multi-column (`t.cols > 1`) so they take the ext-table
   path; the `R<w>` tables (R4/R8/R12/R15/R16) are identity tables and take the graphed range path unchanged.
   Expected from a-gpu2's v1 numbers (1.79 s ×4096 on the H100): v2 well under 1.5 s.
2. The v2 arithmetic bucket is already 4.9x v1 (2 checker layers, depth 1) and the opening 1.5x (109 M vs 178 M
   committed elements); INT8 opening (a-gpu2) shrinks it further.
3. Verifier: 8.1 s Python; a-gpu2's vectorised Ligero verifier (1.34 s on v1) should carry over — the v2 path adds
   only `check_chain_links_clear` and the table union.
4. Field/label: BabyBear^6 throughout.  Every checked v2 gadget is z3-verified over BabyBear (unit body:
   `gpu/v2/results/z3_babybear_v2.json`; epilogue: `lane/b-air-v2` `pack_epilogue_limbs`).  Do **not** switch the
   export back to `BabyBearV2Api("word")` for bench numbers — that reopens the wrap attack (DESIGN §5b).

## Open

* `gkr_export.v2`'s synth / negatives generators still emit the word-form epilogue (their hint paths); porting
  them to `epilogue_limbs` would make the a-v2 chain negatives run on the honest-label circuit too (the 34 cases
  attack the chain and the unit, which are unchanged, so nothing is unverified — it is hygiene).
* The Rust `a-v2` prover's `inst.tables()` takes the first segment's tables: the epilogue's `R16/R15` queries are
  unproven there (DESIGN §5b) — reported, not fixed on this branch (not this lane's file).
* Bit-exact transcript against the CPU prover: not comparable (Goldilocks^3 vs BabyBear^6, different embedding of
  the epilogue); the Python verifier is the same code path that rejects every negative class.

---

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
