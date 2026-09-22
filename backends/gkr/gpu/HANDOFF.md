# a-gpu2 handoff (2026-09-22 ~09:55Z)

State: **prover at 1.790 s median for the full K=1536 VU x4096 on the H100 (v6, run r20260922-094040-9e76; v5 1.857 in
r20260922-092333-c8f0 was the breakthrough entry), 2.5x over a-gpu's 4.43 s, verified 4096/4096 x3; 52/52 negatives
rejected on the v6 path (r20260922-094606-1813); verifier 1.34 s (was 7.1 s).**  Ledger:
`backends/numerical/reports/ledger/a-gpu2.jsonl`.  Branch `lane/a-gpu2` (includes the merge of `lane/a-fusion` 3718a4a).

Buckets v2e -> v6 (s): lookup 1.40 -> 0.43, arith 1.07 -> 0.66, open 1.44 -> 0.53, commit 0.45 -> 0.10, wires 0.07;
verifier 7.1 -> 1.34 s (Python+torch); proof 33.9 MB; depth 438; 1921 transcript slots; peak device memory 47.9 GB.

v6 over v5: `ligero.verify_open` evaluates every functional row at the 192 opened columns as one modular GEMM
(`eval_rows_at`: INTT matrix x Vandermonde -- the same values as encode-then-gather) and `q(eta_j)` as a matmul
(was a 1.6M-op Python Horner loop); every check kept.  Prover: `add_input_claim` as a fused rank-1 kernel.

## What changed (all bit-exact against the a-gpu torch path: same messages, root, opening)

* `gkr_packed.py`: GKR phase 1 on a-packed2's k=3 Montgomery packed sumcheck.  Tables laid out gate-major over
  bit-reversed copies (`_layout`, a Triton gather-transpose); the weight `w(g) x eq(gamma, c)` is the kernel's
  `e_lo x e_hi` tensor product (`e_hi` is arbitrary); the eq windows per round are strided slices of the layer's eq
  table rescaled by batch-inverted prefix/suffix constants (`_EqSlices`).  Same round schedule and messages as the
  torch loop (`gkr.phase1_round`), verifier untouched.  Per full layer 700 -> 93 ms (device ~30 ms; the rest is host
  prep: ~1 ms `_inputs` + ~1 ms kernels per round x 19 rounds).
* `logup_packed.py`: LogUp on a-fusion's `GraphLogUp` with `rounds_per_graph=1` -- one graph replay per round, the
  a-gpu transcript absorbs `S_i` / squeezes `r_i` between replays (`_FSGraph._walk_fs`), `_FSGraphExt` for the
  vector-valued tables (tree + big level from `p_leaf/q_leaf`).  Graphs are built once per (shape) per process
  (~60 s compile+capture, then reused: `gpu.run --warmup 1`); their static buffers stay resident (~20 GB for R16).
  Eager fallback: `VERITY_GPU_NO_GRAPHS=1`; torch reference: `VERITY_GPU_NO_PACKED=1` / `VERITY_GPU_TORCH_ONLY=1`.
* `ligero.py` / `field.py`: opening GEMMs on INT8 tensor cores (`mm_mod_int8`, `Limbs8` with the B operand
  column-major; a-operand written once in int32 into the padded limb layout, fused `split_limbs` kernel); commit NTT
  via b-encode's `RSEncoderSIMT`; `Opening.columns` as a numpy array; `eq_table` for n > 16 as a fused tensor
  product (`eq_outer`, int32 out) -- the n=28 table used to be a 26 GB int64 transient.
* `field.W` is 22 (a-packed2's extension polynomial `X^6 - 22`; a-gpu used 31) -- `logup_packed` asserts equality.

## Where things are

* Pod `vy-g5` (`/tmp/g5.sh` = ssh helper on the laptop; sync = `tar czf - backends/gkr/gpu backends/gkr/packed |
  /tmp/g5.sh 'cd /workspace/verity && tar xzf -'`), `/workspace/venv312/bin/python`, `PYTHONPATH=/workspace/verity/
  backends/gkr:/workspace/verity/backends/gkr/packed`; data `/workspace/bb/pos4096`, `/workspace/bb/neg`.
  Recorded runs: `research run --on vy-g5 --project verity --source . --env PYTHONPATH=backends/gkr:backends/gkr/packed
  --stage gkr.gpu.vX -- bash -c '/workspace/venv312/bin/python -m gpu.run prove /workspace/bb/pos4096 --device cuda
  --warmup 1 --reps 3 --json $RESEARCH_RUN_DIR/run.json'`.
* Helpers on the pod (not in the repo): `/workspace/cmp64.py DIR N` (packed vs torch path, message equality),
  `/workspace/warm.py DIR N REPS` (warm timings), `/workspace/prof2b.py DIR` (synced per-function timers, warm),
  `/workspace/p1chk2.py b,g_n ...` (packed phase 1 vs torch phase 1 at arbitrary shapes).

## Next steps

1. **Open 0.57 s**: `open_w_qc` 0.41 s = four INT8 GEMMs (the two big ones 24576 x 43417 x 4096 at ~55 ms each are
   at ~50% of INT8 peak; the INTT-side ones and the int64 recombination are the rest) + 0.29 s building the Ligero
   functional (`add_lookup_claim` / `add_input_claim`: memory-bound passes over the 26 GB int64 `acc.a`).  Store
   `acc.a` in int32 (halves that traffic and the peak memory; the scatter kernel `scatter_terms_kernel` loads it via
   `load6` -- make the loads dtype-agnostic) and exploit the low-rank structure of the input claims in the GEMM.
2. **Arith 0.66 s**: phase 2 (0.20 s: two `sumcheck_prod` chains of 10-11 rounds per layer, ~30 launches each) and
   `eval_gates` (0.13 s) are launch-bound; phase-1 host prep (~1.9 ms/round) could move into one prepared-inputs
   kernel (a-fusion's `prepare_inputs` / `sumcheck_layer_device` are sync-free for the standard layout).
3. **Verifier**: 1.34 s; the rest is `verify_table` (0.33 s), interpolation / `e_*` Python ext arithmetic over the
   1552 transcript absorbs, `eq_table`; a Rust BabyBear^6 port (`backends/gkr/src/`, a-babybear's `field.rs`) would
   take it well under 0.5 s.
4. **B = 8192**: not attempted -- 47.9 GB peak at 4096 with ~20 GB of graph buffers resident; needs the int32 `acc.a`
   (step 1) first.
