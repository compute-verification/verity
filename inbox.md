[2026-11-14T07:58Z] 325236a2a7a1
RECORD, Fri 13 Nov 11:58 PM PST. **k_by_kind is pushed to #9968: the writer's own commit is 44d883d9106f001e0f110801e5f274db1c693071, and the head is c0a05df8481d0780906343eb410ff641897aed4b.**
- Commit 44d883d9106 touches only `benchmarks/one_stage/inference_only.py` and its test. `session_row` gives `k_by_kind` {`linears`, `scan_state`} from the same count as k: each kind's bits are what `block_counts.fresh` drops by when that kind's registered parameters (Pearl+'s `pearl-y/…` values, and `verity/certifier/scan-state/v1`) are held too. The writer refuses a class whose kinds don't sum to k. `verdict` recomputes the split on the run's circuit and refuses a split that differs or doesn't sum to k (one test each); a policy without the key reads as before.
- RAN, locally on the rehearsal's class (4,27,2, pearl-c-h100-v1-h3, its stated φ): k 337,786,912 = linears 337,641,472 + scan_state 145,440, and with every parameter held 0 bits are free. 145,440 is the scan state row's words × 16.
- READ: the Lean doesn't see the key. `served_record_case.py` takes `--count`'s {table, c, k, n} and `--limits`' alpha and gamma (its docstring, lines 16-17), and never reads the policy document. `inference_only_lean.py` only locates the Lean package.
- Commit c0a05df8481: `record.py`'s `tied_of` reads the short keys. A policy written before the key reads "no figure for the tied class" (tested).
- RAN on this VM: the writer's tests, 31 passed; the benchmark's two test files, 42 passed.
- The target PR is [#10012](https://github.com/compute-verification/verity/pull/10012) at 35a3745cf57. Its run r20261114-074850-779f was launched at 11:48 PM, before the quick-tier ruling, as `--changed` without `--quick`. It stays the one run at that head.

