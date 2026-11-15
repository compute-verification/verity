[2026-11-15T03:31Z] 044fbcd0f128
Lead 03:33Z, to every proofs worker: two of our runs were cancelled tonight at exactly 60 minutes (the queue default for research run --on auto). Any build or suite that may take over ~50 minutes: launch it with --timeout 150 (and check a run already in flight; relaunch a copy now if it has the default). There is no time for a second try before the 05:30Z stop. Re-run your command.

[2026-11-15T03:33Z] b10a9409d29d
Lead 03:41Z: AUDITMAP is moving the audit drivers back out of verity/ tonight (branch cursor/proofs-auditmap-audit-back-3c7e, not yet on architecture): spot/prover/audit/{window_audit,template_audit,check_memo,compose_circuit,prove_bundle,stop_audit,two_stage,prove_path,records_audit} and spot/prover/cost/{class_rows,window_cost} return to benchmarks/one_stage and benchmarks/prove_path under their main names (a5.py, a0.py, prove_path.py, run.sh, ...). If your launcher mends touch or name those paths (job.sh, prove_time.sh, lean_shard.sh, one_stage/a10.sh, a5_llama1b_h200.sh, window.py), fetch that branch and make your mend hold on both layouts, or tell me the conflict. Re-run your command.

