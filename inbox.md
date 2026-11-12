[2026-11-12T20:02Z] 96673709ba67
BACKWARD2, Thu 12 Nov 12:01 PM PST

**I have read BACKWARD's handoff and the branch; both runs are still waiting in the line, and the test of (a) to (c) is written and passes on my VM, committed locally, not yet pushed.**

- Branch `cursor/fa2-train-backward-d232` checked out at `84e7fc9161` (READ: `git log`); merge-base with main `6d95fcc68d`.
- `r20261112-192536-b84b` (suites, pre-mark recipe): state `waiting`, no host (RAN: `research status --json`, named fields only).
- `r20261112-194241-5573` (every word, 64 heads): state `waiting`, no host (RAN, same).
- Step 2 done locally: `test_the_head_at_a_stand_in_shape_keeps_its_signature_and_builds` in `verity/definitions/tests/test_fa2_train_bwd.py`. It pins, at T 16, D 32, BM = BN 16: the stand-in entry `(2, 2, 16, 32): (16, 16)` binds through `train_sdpa_backward` to `Fa2TrainBwdHead_v1{T=16,D=32,BM=16,BN=16,NS=47}` (STEP's `TINY_SDPA` and the grader conftest's `SDPA`, READ on `cursor/trn144-step-fused-attention-2556`); the direct bind at NS = 1 (the frontend's structure stand-in, READ there); both heads' signature, `(do, q, k, v, o, lse)` with five `Array(16, Array(32, BF16))` and `Array(16, F32)` in, `Tuple` of three such out; the `_v2` head builds on bits; and on words both equal the kernel's loop written out in the test. RAN locally: 1 passed in 2.0 s; the file's quick tier 5 passed, 1 deselected, 7.2 s; ruff clean. No Definition, signature or table shape changed.

What I understand is left, in order:
1. Read the every-word run when it ends (TOTAL lines of `out/fit.txt`, `fit.json`); if a deterministic head misses a word, I tell you at once.
2. Read the suites run when it ends; a failure naming this change's files is mine, one the base shares is not.
3. One commit: the every-word run record as `"fit"` in `verity/definitions/evidence.json`, `CONFORMANCE` citing `_FA2['fit']` as the forward's cites its fits, and the test above (I will fold it into that commit before pushing). Plus any TILES MUST FIX you pass me that touch no Definition.
4. At that head: the pre-mark suites (`--changed <merge-base> --quick`, 8 CPUs, 32 GB as last time) and circuit-check on both sidecars (8 CPUs, 48 GB) on nodes, avoiding cpu-2.
5. The PR body, from the current one, keeping its Lane, Refs and Goal lines.

Not mine: the masked scores, default mode, the frontend rule, the grader.

