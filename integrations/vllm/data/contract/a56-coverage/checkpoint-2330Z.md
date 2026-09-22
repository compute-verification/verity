# A5/6 checkpoint 23:30Z — F2 fixed (builder-side), `for-a7.md` written; no pod tonight

Mainline `1ca0a9e8` merged; directive `cursor-directive-2026-09-10-1604.md` §2/§3/§5 read (no O1/O2 choice from T=9/65; next measurements n=64 batches, table-free guest, O2 packed path; labels stay attached). (a)–(d) wait for the extension.

## 1. F2 fix — `zk/bench/attn_gates.py`, lowering-side only

Cause: in a FIRST block a lane holding one column after a multi-column lane (NVIS ∈ {3, 5, 7}) had its sum = the P gate itself, emitted before the previous lane's add → outgoing not in gate order → `_lower_subcircuit` refused. Fix: that lane is materialised as `f32_add_ftz(p, ZERO32)` (exact: p is an ex2 output ≥ +0 / +inf / NaN; the same materialisation the empty lane already uses). Lane 0 with one column (NVIS = 1) is untouched.
Evidence: all 256 (D ∈ {64, 128}) × (NVIS 1…64) × FIRST shapes lower and have 1+4+D strictly increasing outgoing gates; raw-structure and lowered digests of the **250 previously-lowering shapes are byte-identical** to the pre-fix table (`/tmp` comparison at the fix), the 6 previously-failing shapes gained a digest (no digest *changed*, so not PROPOSED); value equivalence `p + ZERO32 == p` on 120 random blocks; forced heads on the real TINY record at t = 2, 4, 6 (layer 1, head 2) → covered / neighbour refused / fa2 refused (`records/forced_head_run-TINY-t2-L1-h2-after-F2.json`). Regression test `tests/contract/a56_test_head_builder.py::test_f2_every_attn_block_shape_lowers_with_outgoing_in_gate_order` (11.8 s); head/locator/coverage tests 17 passed. No partition, record, or approved-guest digest is involved (the head kind digest is builder-side, computed identically on both sides).

## 2. `for-a7.md` (+ `for-a7-populations.json`)

Per option × workload: populations (O2 = as-built N_V 103.03 G / 166.2 M; O1 = 4.65 G / 161.9 M with 2 186 730 / 12 690 head units), widths (scalar VUs one 32- or 16-bit word; head D×16-bit words, (2T+1)·D inputs), unit prove costs MEASURED (approved n=1 core 73.6/55.0/33.7 s = $0.018/0.013/0.008; fa2 head T=9/65 core 623.6/860.0 s = $0.15/0.21; compressed ‡) and NOT MEASURED (FTZ five as scalars — not expressible in fa2's 16-bit `In`; n=64 amortisation; large-T heads — EXTRAPOLATED by rule ≈ 3 900 s ≈ $0.95 at T≈800, labelled; table-free guest), per-RU commit (O1 0; O2 389 GB/RU MODELED, 244 s = payload + frame hashing only — no encoding/tree/storage/openings/RSS), and the rest bucket: CK-ELEM's, unbuilt because its operands are uncommitted norm/argmax interior — offered as a third column "norm interior" (2.47 G / 148.9 M per RU; dominates 32×15 at 89.6 %).

Stopping, per 00:01Z.
