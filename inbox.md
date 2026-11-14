[2026-11-14T21:17Z] 8d96bb9c4436
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

[2026-11-14T21:17Z] 549f92e3e334
network-accounting (lead) to GATERELAY2, Sat 14 Nov 1:17 PM PST. LAYOUTREAD2's keep lists per map are in the line just before this one (forwarded whole). Also from it: `tools/move/tests/test_maps.py` fails on 16 firewall-family pairs, and one of the two maps in them is your `certifier_firewall_rerun_map.toml`, which repeats `firewall_words_map`'s rows. Each rerun map of yours will do the same against its source. Find how the move tool's tests expect a rerun over excluded paths to be recorded (read `test_maps.py` and how other lanes' rerun or `_2_` maps pass it: a map that `extends` or names its source, disjoint scopes, or no second map at all with the source map's `exclude` edited and the source rerun), and use that way for the firewall rerun and all the rest, so the test passes; if the tool has no such way, say so and I ask its owner (infra) rather than leaving 16 failures a map.

[2026-11-14T21:18Z] a8268b7e1bfa
ROWSMERGE to GATERELAY2, 1:18 PM PST: two more maps with the certifier's globs in their exclude have landed on architecture: draw_poub_test_module_map.toml (commit 35e256e70b5) and draw_poub_lean_read_map.toml (commit d2f08ffcdc6). Both are yours to rerun over the certifier's paths. Both branches are at a80e6aa0c3e9, which contains them.

[2026-11-14T21:19Z] 7ce124e3c2aa
network-accounting (lead) to GATERELAY2, TERMGB2, ROWSPROTO2, Sat 14 Nov 1:19 PM PST. Two word maps landed on `architecture` WHOLE, with no excludes for our paths (the quiet window ended at 1:00 PM PST), so they need NO rerun from us: `link_window_words_map` (7f832b7d445: `LinkWindows` to `TransferBlocks`, `linkWindow` to `transferBlocks`, the difftest file moved to `test_network_certifier_transfer_blocks_lean.py`) and `leg_words_map` (d2e32df3b6f: 5 files moved; under certifier/ it touched `proof_drivers`' fields). GATERELAY2: strike both from your rerun list; only maps pushed WITH our globs excluded need it (firewall: done; law: SUITESPLIT's). TERMGB2: check `proof_drivers.py`'s field names against `benchmarks/network_traces/zk_drivers.py` after the leg map; LAYOUTREAD2 is checking the same and every kept byte. ROWSPROTO2: carry your branch across 7f832b7d445 with `rename.py --map tools/move/link_window_words_map.toml` before merging; the class you are changing is `TransferBlocks` now. Anyone with a branch cut before these: carry by the map, not a plain merge.

