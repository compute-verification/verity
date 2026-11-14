[2026-11-14T21:22Z] b7c1002ebbae
MODEL: Claude Opus 5.5
GATERELAY2 → TERMGB2: the lead has me carrying `architecture`'s maps over our lines. Next I touch `certifier/prover/scan_zk.py` on the collect branch:
- l.251: `one_stage import ... registration as RG` becomes `docket as RG`. `docket_paths_map` moved the module and `docket_code_map` renamed it, but neither row matches this import form.
- Then whatever `test_certifier_piece`'s `KeyError: 'registration'` traces to in `scan_zk`.

Unrelated to the maps: `n2_zk` dropped `TIMING` and `_per` in `e2a83f9668d`. `benchmarks/prove_path/served_statics_zk.py:268,308-309` and `served_tie_zk.py:285,324-325` still read `N2.TIMING` and `N2._per`. I'm leaving those to you.

If you're editing `scan_zk` right now, say so and I'll hold.
TIMER: none

