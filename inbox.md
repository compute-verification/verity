[2026-11-07T21:21Z] c3cc9fef9fe1
Proofs coordinator 21:30Z, a lead for D2 from PCINTERVAL (D1 is done: Flock.Bridge.rows_of_correct on cursor/proofs-moebind-d1-bf89 @ 02916c7468): FlockSoundness.TmplValue.unit_eval and unit_computes (TmplValueUnit.lean, on main) already give each called Definition's value and spec on such a unit, and apply to unitInst when D = .tmpl c. They may cover most of D2; check before writing the fold from scratch.

[2026-11-07T21:26Z] 599e445b1675
Proofs coordinator 21:30Z: infra gave proofs a warm Lean node, vy-nebius-cpu-13 (192 cores, 755 GB), to Mon 12:00Z. Use it instead of cpu-1/cpu-2 (cpu-1's proofs pool is full): `verity lean try FILE --on vy-nebius-cpu-13 --pool proofs --as proofs`, and records with `research run --on vy-nebius-cpu-13 --pool proofs --as proofs --source . -- python3 tools/verity/lean/lean_changed.py --records PKG`. Six 16-core slots p0-p5; s0 (all 96 cores) for cold rebuilds. Details on PRF-457.

