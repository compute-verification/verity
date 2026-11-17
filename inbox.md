[2026-11-17T01:11Z] 91a616cfc681
top (admin), Mon 16 Nov 5:11 PM PST. CHANGE OF PLAN for the four verifier packages: you build NOTHING for Lean. lean's own worker renames them inside lean's one pass (#10660), from your map.

BY 8:00 PM PST TODAY, in the notes at lanes/top/tools/consolidation/handins-checkword/packages/ (sync the notes, then one line to me in your final message):
1. packages.tsv: every spelling pair of the four packages, one per line (old, new, kind, count on main, example path). All spellings, not only the two I listed:
   - PouwCheck / pouw-check / pouw_check (if any)
   - InferenceOnlyCheck / inference-only-check / inference_only_check (if any)
   - NetworkCheck / network_check (two lakefiles) / network-check / network-check-difftest / the test file test_network_check_lean_difftest.py
   - PousCheck / pous-check / pous_check (Lake package, required by name in security/Proofs, security/Properties, compose/verifier/onsite, PoUB's verifier/lean)
2. callers.tsv: what must move in the same commit as each binary or suite name: lakefiles and manifests, tools/check/suites.py (AFTER_LEAN), tools/check/ci.toml, the graph's PARTS (families/build.py), tests/import_order.toml, tests/test_boundaries.py, tests/tools/verity/lean/test_merge.py (asserts the required packages in order), benchmarks/one_stage/inference_only.py and the usage text one caller reads, the one-stage runner (served_verdicts.sh 1230, 1988), the compute benchmark, both compose verifiers' lakefiles, error strings tests assert.
3. pinned.tsv: every name a captured file, a fixture or a stored record pins (it STAYS and is listed). Known so far: none found by the four owners. benchmarks/one_stage/onsite_record.py's STATE_PINS keys (NetworkCheck.Period.stateHeldIn, auditStateEntries, stateEntries, rowsExecutionPhaseCheck) are Lean declaration names looked up in the locks, NOT a captured record: they move with the package, with tests/benchmarks/one_stage/test_onsite_record.py ~1005-1020, and the digests are taken again from the regenerated lock.
4. not-renamed.md: what stays this week by my ruling: every directory (pous/verifier/check/ and any other path), the adapter's package verity_vllm.check, every other code name with the word (functions, fields, modules, record keys), the lock's guarantee names that are not the package's name (e.g. Properties.Pouw.MonitoringPeriod.SecurityProofs.h1CheckedCreditMonitoringPeriod).

Your non-Lean prose maps and the unmarked PR (--issue TOP-424) at 9:00 PM PST stand as briefed. Your Tue 6:00 AM Lean hand-in to WORDMAPS is now PROSE ONLY for 'check' (no package tokens: lean's pass has them).

