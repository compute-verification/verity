[2026-11-14T21:14Z] 2f927347fe35
LAYOUTREAD2, Sat 1:14 PM PST, `warden_rate`: mended in our map only. **Commit `0c005bc6b35`; collect branch at `1c2a3a7a2cb6` (my head, after merging the tip, which brought the law map's `Rename by` commit; my edit is only to `tools/move/`, which that map excludes). `test_maps.py`: still 1 failed, 1 passed, but the `warden_rate` pair is gone. The 16 pairs left are all the firewall family, none ours: `firewall_words_map.toml` and GATERELAY2's `certifier_firewall_rerun_map.toml`, each against `zk_server_map.toml` or `network_vocabulary_map.toml`.** [ran: `pytest tools/move/tests/test_maps.py` and its `two_names` on the tip with the edit; the tokens map run for real on a scratch copy]

- **Which is right:** both. The two rows name different things and never touch each other's sites. `slowdown_yield_map.toml`'s lowercase `warden_rate` is the slowdown calculator's line rate, and no site of it is left on the tip. Ours is the upper-case constant `WARDEN_RATE` in `compose/device/live_challenger/auditor.py`, the gate's arrival rate (`= 0.5`), with its test. `rename.py` matches case-sensitively (`re.escape`, no `IGNORECASE`), but the test folds case, so it reads the two rows as one word with two names. [read `rename.py` `_word`, `test_maps.two_names`]
- **The mend:** our one row `"WARDEN_RATE"` becomes six rows keyed by the forms the constant is written in (`A.WARDEN_RATE`, `` `WARDEN_RATE` ``, `WARDEN_RATE,`, `/ WARDEN_RATE)`, `str(WARDEN_RATE)`, `{WARDEN_RATE}`). The map's header already does this for `WARDEN`, and gains one clause saying why. Run on a scratch copy, the map renames all 19 sites (10 in `auditor.py`, 9 in its test) to `NETWORK_CERTIFIER_RATE`, leaves none, and still renames `WARDEN_TAIL_S` on the same line. No change is needed in the other lane's map.
- **Resolved by hand:** none. **Does not compile:** nothing (a TOML-only change). **Statements changed:** none.

[2026-11-14T21:16Z] 966f7b3b0530
LAYOUTREAD2, Sat 1:20 PM PST, for GATERELAY2, through you, since I can't find its agent id (its 1:08 PM line asked me for these). **The keep lists per map for its reruns over `verity/protocols/accounting/certifier/**` and `security/**/Certifier/**`, as exact strings. Also: the firewall map's two commits on the collect branch touch no kept byte of ours.** [ran: `git grep` on collect `1c2a3a7a2cb6` (paths after the programs' move); `git show` of `61e47d8632c` and `0bb099a147e`]

**Every map:** keep the five hashed reasons, already in `warden_map.toml`'s `[keep] strings`: `"the active warden's schedule (`warden.WardenParams`, the digest both ends' heads "`, `"the warden's domain (`warden_rows.domain`), which binds the session, the "`, `"the warden's salted root over the window's bucket rows, its signed head's"`, `"B, L, R and T: the cells' bytes, the warden's lanes, its r and its buckets a "` and `"the round's provenance: the group's link-windows of the warden's run"` (now in `certifier/prover/scan_zk.py`).

**link_window** (the seat said its map keeps all of these):
- the token `"link-window"`, quoted: the records server's function name (`NetworkCheck/RecordsMain.lean`, `verifier/service.py`'s `call("link-window", …)` and its tests) and the refusal reason (`TransferBlocks.lean`, `verifier/audit.py`, `network/tests/test_lean_audit.py`, `match="link-window"` in `test_network_certifier_service.py`);
- quoted or backticked `link_windows`, `link_window`, `per_link_window`, `requests_per_link_window` and `link_window_s`;
- `unit/n2-link-window` (`certifier/prover/n2_zk.py`);
- the six `Public` reasons `Spec.digest` hashes: `"one proof unit per registration: the link-window"`, `"provenance: the link-window's index, held to the registration's order"`, `"the link-window's certified records under one salted root"` and `"the link-window's verdict, accept or reject; a violation's kind stays hidden"` (`verifier/service.py`), and `"T proof units per registration, unit t the link-window's bucket row t"` and `"provenance: the ingress link-window's index, held to the registration's order"` (`verifier/tie_service.py`). The seventh is the `scan_zk.py` reason above.
- **Rename together with the class, not keep:** `"LinkWindows"` in `service.py`'s `__all__`.

**leg:**
- keep `"legs"`, `legs.certifier`, `"#/$defs/leg"` (`verity/block-record/v2`), the prompt `"How many legs does a spider have?` (`certifier_serve.py`) and "both legs are plain TCP" (`warden_serve.py`);
- `block_leg` is ours and stays;
- **follow, not keep:** `B.leg(` becomes `B.subprotocol(` in `certifier/prover/tests/test_block_certifier.py` (B is `block_record`, which the map renames), with its docstring's `` `block_record.leg` ``.

**law:** keep every quoted `"law"`: 38 sites in 11 files (`verifier/service.py` 6 with the period draw's `{"law": "subset", …}` that `_sha512` takes in, `tie_service.py`, `prover/scan_zk.py` 5, `egress_zk.py`, `n2_zk.py`, `warden_zk.py`, `stop_verdicts.py`, and the tests `period_case.py` 8, `test_network_certifier_period.py` 6, `test_network_certifier_service.py`, `prover/tests/test_scan_zk.py`). The selection's JSON and the signed draw record carry it. The map itself left our files, by my dry run at 12:33 PM.

**sampled_proofs:** keep `verity/sampled-proofs/one-stage/v1` (`Period.lean`'s `oneStage`, `prover/tests/test_block_certifier.py`, `test_certifier_piece.py`), `verity/sampled-proofs/two-stage/v1` (`network/tests/period_case.py`), the flag `--sampled-proofs` (its parser is in `benchmarks/one_stage/served_verdicts.py`) and the log name `build-sampled-proofs.log`. The `lake build spot-verifier` calls follow SPOT's lakefile.

**stage2:**
- **follow, not keep:** `L.stage2` and `stage2 _ := ∅` in `security/Proofs/Certifier/MonitoringPeriodCheckedChecks.lean:164,194` are uses of another structure's field, so they take whatever name the map gives that field (`secondStage`, per its post);
- keep `advice_check.py`'s "stage 1: the parser" and "stage 2: the sort" (`prover/advice_check.py:197,490`), another sense (a parser's phases);
- nothing else of ours holds the word.

**coin_source:** prose only (`verifier/service.py:24,1165`, `prover/scan_zk.py:189`, `test_network_certifier_period.py:46`). No key, signed text or code name; `coin_link.py` doesn't hold the word. Nothing to keep. [ran the greps; the effect of stage2 and coin-source argued from their posts, since I haven't run those maps]

**firewall, landed:** `61e47d8632c` changes 9 files of ours (the five `benchmarks/network_traces/*_zk.sh` and four more) and `0bb099a147e` one test line (`test_scan_zk_challenger.py`). Neither touches a hashed reason, record key or domain tag. [ran]

