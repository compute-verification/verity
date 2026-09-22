# a-verifier handoff (2026-09-22 ~09:45Z)

State: **deliverables 1-3 done**; 4 (checker v2) not started.  Branch `lane/a-verifier`, everything committed.
Nothing is running on vy-cpu2 or vy-g5 for this lane.

* Verifier: `backends/gkr/verifier/` (README.md).  Build: `cargo build --release` (no deps; `.cargo/config.toml`
  sets `-C target-cpu=native`, ~2x on the EPYC).  On vy-cpu2 the binary is `/workspace/cargo-target-a-verifier/
  release/verity-gkr-verify` (also copied to `/workspace/verity-gkr-verify`, older build).
* Proofs + statements on vy-cpu2: `/workspace/averifier/proofs/r20260922-091240-4bf9/{stmt,proof4096_torch.bin,
  proof4096_kernel_b.bin (rejected by both verifiers), negproofs/<id>.bin}` and `.../r20260922-090318-2864/
  proof64_torch.bin`.  Same files on the laptop under `~/.research/runs/<id>/`.  `stmt/` = circuit.txt, epilogue.txt,
  manifest.json, public.bin, neg/<id>/public.bin (no witness files).
* Generating proofs (vy-g5, `research run --on vy-g5 --source . --env PYTHONPATH=backends/gkr`):
  `VERITY_GPU_TORCH_ONLY=1 /workspace/venv312/bin/python -m gpu.run prove /workspace/bb/pos4096 [--vus N] --device cuda
  --proof-out $RESEARCH_RUN_DIR/proof.bin`; negatives: `... negatives /workspace/bb/neg --device cuda --proof-dir
  $RESEARCH_RUN_DIR/negproofs`.  **Without** `VERITY_GPU_TORCH_ONLY=1` the committed kernel path's proofs are
  rejected by both verifiers (`a_verifier_discrepancies.md` F1) -- the a-gpu lane has to commit its pod working copy.
* Results: `backends/numerical/reports/a_verifier_discrepancies.md` (findings, table, e2e), ledger
  `backends/numerical/reports/ledger/a-verifier.jsonl` (3 entries), envelopes `results_e2e_{kernel,torch}.json`.

## Next three steps

1. **Sub-second verifier** (3.0 s at 12T now, 93% in the Ligero linear test = 43417 rows x 4 NTT_4096 + the `a`
   scatter): pruned forward NTTs (only ~64 outputs per coset are read), AVX-512 butterflies (interleaved six-lane
   layout won; planar lost), cheaper R16 flush in `Functional::unit_row`.  Measure with `--json` (`t_rows_fill_cpu`,
   `t_rows_eval_cpu`).  Or port the linear test to the GPU (rows are independent).
2. **checker v2** (`a-v2`, PROTOCOL.md §14): `circuit.rs` parses the generic text format; add the v2 gate/query kinds
   if any differ, then run against an a-v2 export (main) -- the transcript, LogUp, Ligero code is shared.
3. Re-verify a kernel-path B = 4096 proof once the a-gpu lane's working copy is committed (their `run.py` there has no
   `--proof-out`; merge `lane/a-verifier`'s `prover.py::Proof.to_bytes` first), and re-record the e2e row with the
   real prover time.
