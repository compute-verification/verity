<decision>
- **The tree.** The run branch `cursor/circuits-cir272-bound-6f04` is `main` 5f0c008aa040 merged with 31b1b56aec2a (for `stage.py --bind`), plus `scratch/cir272/`.
- **The runner.** `bound.sh` is CIR-259's (the 9dd0 version, with the stager's memory stop, plus 1147's Lean-peak rule). I added:
  - each type in its own process group under a deadline;
  - a budget that lists the types it can't reach rather than starting them;
  - inputs read from the tree rather than `$RESEARCH_RUN_DIR/inputs`;
  - threads held to 8.
- **`others.py`.** Its "other Definition" is now one of main's card names, so it resolves on the merged tree.
- **The lists.** `make_types.py` builds them from the CIR-271 v2 record, largest first, dealt alternately. The first run took 4 types, to show the tree works.
- **Set aside:**
  - falling back to 31b1b56aec2a, which wasn't needed: every name resolved;
  - fast types first, because the brief says largest first;
  - dropping the negative controls, kept to match CIR-259.
</decision>
<risk>
- About 50 of the 240 types are at k_log 27, and two 6-hour runs won't reach most of them. Expect many "not reached" entries.
- The 2,700 s per-type limit may cut k_log 27 types short.
- The prover's and Lean's own parallelism beyond the RAYON and BLAS variables isn't held to 8 cores.
- The OOM killer, if the 62 GB stager watchdog is too slow, could hit something other than the stager.
</risk>
<verified>
- `bash -n` on `bound.sh`.
- `make_types.py` asserts 240 types and 2,998,736,728 blocks.
- The first run, r20261112-214915-789b on cpu-15, ended rc 0. It accepted 4 of 4 types: `B.verify.held` was true for MoE SWAP=false, BatchedGemm K=192, SparseMlaHeadEnd DV=64 and ResidualAdd N=384.
- Both batch runs started and resolved all 127 names, apart from StopCheck's known refusal. Their first types finished.
</verified>
