# a-fusion lane -- HANDOFF

Branch `lane/a-fusion` (worktree `/Users/danielreuter/projects/verity-a-fusion`), pod `vy-g4` (H100, id uktx6i8dejuunk).
Pod working copy: `/workspace/fusion` (synced with `tar | ssh`, helper scripts `/tmp/g4sync.sh`, `/tmp/g4ssh.sh` on the
laptop: ssh `-i ~/.runpod/ssh/runpodctl-ssh-key -p 11461 root@64.247.201.61`). Kernel cache `PACKED_KERNEL_CACHE=/workspace/fusion/kcache`.

## State (2026-09-22 ~01:30 PDT)

Done, committed, tests green on the pod (`PACKED_DEVICE=cuda pytest backends/gkr/packed/tests`):

1. **LogUp CUDA graphs** (`logup_graph.py`, `kernels_graph.py`, tests `test_logup_graph.py`): v1 shape 427 ms eager ->
   **88.9 ms device / 92.0 ms wall** (per-level graphs, coins pre-filled) or **103.5 / 107.1 ms** (per-round graphs, Fiat-Shamir
   sync between rounds). Result `backends/numerical/reports/a_fusion/logup_graph_v1.json`. Ledger component entry added.
2. **Packed sumcheck fusion** (`kernels_fused.py`, `kernels_triton.py::_packed_u_source / _ext_fold_ip_source`,
   `sumcheck_packed.py`): U pass (k packed rounds from one read of g, h) **3.48 ms vs 7.83 ms** for the three packed
   rounds at 2^28 k=3 (generated explicit-name kernel; the tensor-broadcast twin is 13.7 ms -- kept for the agreement
   test only). Fold+Hadamard fusion (`ext_fold_ip`) **measured slower** (2.28 vs 0.86 + 0.88 ms): SIMT-bound both ways;
   opt-in `Kernel(fold_fusion=True)`, default off. `prove_device` + `prepare_inputs` = sync-free pipeline (whole
   sumcheck in one CUDA graph) for benches / a-gpu.

Running on the pod (nohup chain, logs in `/workspace/fusion/*.log`, outputs `/workspace/fusion/out_*`):
- `bench_fused --log2 26 28 --ks 3` -> `out_fused/bench_fused.json` (`fused.log`)
- `bench_fused --log2 26 --ks 0 1 2 --no-pieces` -> `out_fused_ks/` (`fused_ks.log`)
- `logup_graph_bench --suite v2 v2-splitu` -> `out_v2/logup_graph_bench.json` (`v2.log`)

## Next three steps

1. Fetch the three outputs (`/tmp/g4ssh.sh 'cat /workspace/fusion/out_fused/bench_fused.json'` etc.) into
   `backends/numerical/reports/a_fusion/`; ledger `component` entries for the fused sumcheck (2^28 k=3 graphed ms/layer)
   and LogUp v2 (per table instance, SHIFT committed vs split-u).
2. Re-price A: extend `reprice_a.py` (or a `reprice_v2.py`) -> `backends/numerical/reports/a_kernels_h100_v2.md/.json`
   with v1/v2 buckets (sumcheck from `bench.layer_model` fit with the fused per-layer ms; LogUp measured; commit/opening/
   hash via `explore.tensor` with `VERITY_CHECKER_VARIANT=v2`), ledger `vu` entries (`--breakthrough` if < 4e6), plots.
3. `kernels.py`: entry points `packed_u_inner_product`, `round_from_U`, `ext_fold_inner_product`, `prove_device` +
   test vectors (`tests/vectors`), README rows ("For a-gpu"); final message per brief 2.9.
