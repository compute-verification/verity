<decision>
- **Retire the types rather than alias them.** `stage.py` refuses each of the 18 `PARTS` names with what to stage instead (`RETIRED`, with `RETIRED_EITHER` for the two that have alternative replacements). I set aside a silent alias to the replacement, because the replacement is another Definition with another pin. A caller that asked for the old name would get a different statement without knowing.
- **Bound by default, `--unbound` as the escape.** I followed the coordinator's decision 1: `--bind` and the `+bind` arm go, and `--unbound` reproduces main's no-flag staging byte for byte. The two callers that hand Lean neither file (`steady_node.py`, `flock_sessions.py`) pass `--unbound`, so their pins and verdicts don't move. Moving them to hand the Program is PRF-567's work, not this PR's.
- **The verdict each caller reads.** `zk_cost.py` and `served_units/summarize.py` previously counted the no-Program refusal as "checked". With a Program to hand, they now hand it and count only Lean's accept; the selftest build's statement is expected REFUSED by `servedForm`. I kept the refusal path in `summarize.py` for `st-honest` rather than dropping that case.
- **`phase_profile.sh`.** It defaults to the census's decode gate-up block (decision 2). It gained `SM` because node 2 is sm_120 and the build defaulted to sm_90 only. The pins were re-taken in one GPU run (decision 3).
</decision>
<risk>
- **`MAIN_ZK` and `MAIN_M0` were taken on sm_120;** main's old pins came from an `SM=90` build. The pins hold elsewhere only if the prover's transcripts are device-independent, which `gpu_proofs_match_cpu` asserts but no sm_90 run of the new default tested.
- **No Lean accept at a full statement's count.** Every Lean accept is at N = 1 or 2; the full W13 statement ran on the GPU without Lean.
- **`window.py`'s `KEEP`** keeps `partition.json` and not `program.json`. Anything that later verifies from a kept window would lack the Program.
- **`--unbound` was compared with main's staging for fp8-coord, SiluMul and fp8-coord `--public-weights`,** and the head block's unbound pin against the steady spec. Other types rely on sharing the code path.
</risk>
<verified>
- Suites `--changed 5f0c008aa040`, without `--quick`: 9 of 9 at 4dff5ca9743 (r20261112-201917-61b8) and 9 of 9 at 9b280baba610 (r20261112-234331-2cc5), none cached.
- parts.sh (r20261112-201828-47ac, cancelled after the head block's first staging at 212 GB):
  - `--unbound`, main's and 31b1b56's `stage.py` give one `circuit.txt`, `inst.bin` and `pub.bin` for fp8-coord K=256 and SiluMul_v2 I=256;
  - bound SiluMul: Lean derives units, accepts with the Program held, and refuses another Definition's Program and a missing Program;
  - served_units job: as expected;
  - zk_cost: accepted 1 of 1;
  - the head block's `--unbound` pin equals glm53-flock-steady-head-block-h200.toml:36.
- pubw.sh (r20261112-202406-85e0): `--public-weights` `--unbound` equals main's. `--packed` fails on main too.
- GPU (r20261112-233111-2abf, vy-nebius-2): W13 staged at 2,048 units; deterministic proofs and transcripts under both provers; gpu_paths_agree passes. The new digests were committed in 9b280baba61.
- Ratchet gate at the head: 0 failed. `git merge-tree` is clean against origin/main 7e70df68bd18 and against lean-remote-merge-onmain-fc8d 67b2c6791982.
</verified>
