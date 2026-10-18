# lean-check-time checkpoint (delete before the PR)

- 23:55Z: step 1 data gathered. Scripts in /tmp/lt (list_checks.py, fetch_lean.py, analyze.py, keys.py, why.py): attempts
  of tool `check` from the store's remote (attempts/r20261017*, r20261018*), run_files manifests, then
  lean-audit/audit.json, audit-*/report.json, steps.jsonl. Twins collapse to the one that ran lean-audit.
- Window: 20 launch groups r20261018-085009-409b .. r20261018-175051-b5ac.
- lean-audit step median 348 / p90 1279 s; 511 / 1730 when the Security trio missed (12/20), 3 s when reused.
- Miss reasons (keys recomputed at each commit with that commit's lean_audit.py): Security trio 8 input changes,
  4 "pass existed on another node, not reused" (overlapping trains: the earlier check still running).
- PR #2599 (lean lane, cursor/verity-lean-facts-fc8d) already overlaps the spec's tail with the prover wait and
  parallelizes Facts.lean pins/canon. Don't duplicate.
- Step 2 candidate: drop from every key the auditor files the audit never reads (lean_changed.py, merge.py, run_daily.py,
  tool.py, pyproject.toml); 4 of 19 landings in a week touched only those and re-audited all 29 packages.
