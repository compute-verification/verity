[2026-11-12T21:47Z] 7dda7f779ad2
BACKWARD2, Thu 12 Nov 1:46 PM PST.

**The probe is ready to launch on your word (vy-nebius-2: one job of about 20 minutes, one card under a lease of at most 5 minutes and 16 GB), and shoulds 4 and 5 are pushed at 31fdc1c38b with no change under `verity/definitions/registry` since 1178eb5439.**

Probe:
- RAN r20261112-213411-af3a (CPU node, smoke mode, no lease): the venv step installed torch 2.14.0 and numpy 2.5.4 in 59 s, and the probe script made both cases' draws and wrote its archive; rc 0.
- Cases (RAN on the VM, numpy twin): `ftz` (dO times 2^-120) has 18,632 of 65,536 dQ words that tell the dQ atomics' flush-to-zero apart. A subnormal dP-sum case tells FFMA from FADD on 0 words at 2^-66, 2^-64 and 2^-62, so `ovf` replaces it: under FFMA the dP sum is a finite 2^127 on all 1,024 rows, and under FADD it is inf on every row, so every dQ and dK word is non-finite.
- After the probe: a CPU fit of its 8 heads with the same `fit_bwd.py`, and its ids go into `evidence.json`.

Shoulds 4 and 5 (31fdc1c38b):
- 4: the head test's inputs cancel across a key's query blocks (query block 0 repeats block 3's q and O with dO negated; block 3's dO is times 2^8, the middle blocks' times 2^-12 and k times 2^-12, so every visible P rounds to bf16 1; LSE is 0). The negative control asserts that the loop with the query blocks first to last gives other dK and dV words (RAN locally: 30 dK and 33 dV words differ). At BM = 16 a block is one k16 step, so the steps' order is should 5's.
- 5: the fixture's key block is now head 3 of `bwd/l16/det/unit`, whose dK and dV each have words that change under each reversed order. The slow test asserts both, and passed in 29 s (RAN locally). Re-registered and preserved as art:253bd5f4d35790a0f729bb2be897f0a72ca690e9a78c18b3aafa28b1c7d0a494, with the old id under `previous`; 227 KB.
- Mutations (RAN locally, every test including slow, one mutation at a time). Each is caught by exactly one test:
  - M1 (the head's query blocks ascending): the head test.
  - M6 (dP sum by FADD): `test_the_dp_sum_is_fmul_then_ffma`.
  - M7 (no FTZ in the buffers) and dropping the first add: `test_a_dq_part_s_add_into_its_buffer_flushes_to_zero`.
  - M9 (Key's and KeyFirst's k16 steps reversed): the slow key-block test.
- Since the Definitions are unchanged, the every-word run r20261112-212456-3c41 at 1178eb5439 stands, as you ruled.

Checks:
- RAN r20261112-212601-33d9, circuit-check on both sidecars at 1178eb5439: "circuit-check: 16 target(s), 0 failure(s), 14 unchecked (warnings, which never fail) on 14 Definition(s)", rc 0. INFERRED: it holds at 31fdc1c38b, which changes no Definition or sidecar.
- Suites: r20261112-212535-7f3f (at 1178eb5439) is still running. r20261112-214623-31a3 (at 31fdc1c38b, `--changed` the merge base, `--quick`, 20 GB) started on vy-nebius-cpu-6. You get the last lines by 6 PM.
- The old-head runs r20261112-194241-5573 (every word) and r20261112-192536-b84b (suites) are still running. I read each when it ends.

