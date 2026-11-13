Both branches now hold `main` at `1e4d5c993c08a3ceee881d8c8563246b42243ea1`, which has #9375.

- #9318: `05dc68622765922ee340657e0fad2d206f6fc80e`. It merges that `main`, with one conflict in `test_dsa_topk_reads.py`, closed as main's slot line (`{IF.PACK_LOG} == {22}`) with our `max(k["DsaPositionsMerge_v2"]) <= 26`. That test file passes: 40 passed, 1 skipped (`r20261113-033216-49da`).
- #9371: `a62164a32a747cae49c7fc46222994c685ccb9f6`. It merges #9318's head cleanly.

The merges moved no pin, no generated file and no `STATEMENTS`:
- The size record's own tool rewrote it byte for byte at both heads.
- The long-row and `dsa_topk_reads` sidecars passed at #9318's head with every pin matching (`r20261113-034020-4471`).
- `circuit-check --all` at #9371's head: 17,340 targets, 0 failures. The shards are `r20261113-035118-af3d`, `r20261113-035153-b2c7`, `r20261113-035206-a1cf` and `r20261113-035219-058a`.
- The ratchet gate gives `failures: []` at both heads.

The suites runs, `suites.py --changed 1e4d5c993c` whole, are `r20261113-033454-e466` (#9318) and `r20261113-035051-a369` (#9371). At 06:02Z each had 48 of its 60 suites passed and none failed, with `repository` at 392 passed. The 12 suites not yet read include `verity-vllm`, which holds the long-row files' slow tests.
