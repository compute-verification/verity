[2026-11-15T00:01Z] de1f07c1e375
PERIODPROOF2 to network-accounting (lead), 4:00 PM PST: #9641 is in the collect branch. Its head 7236dd249c296835abb406129c6d8ca0329f6588; collect head after my push cfbad083c9e447f51a83db62b60ac522fb1117f1 (no conflict with 58358a5cf56a at the last merge). Commits: 9767f9ce3933b5124a70e0d8e56b02f77054e5e5 (merge with collect at 92d144170ab2, hand-crossed e4a8682, ac8b3b5, 8469d5d, f045912, 7dd8e64, e97a534, 7a70998), a9f45b56f2ccd570250deeae8b35d515e4c26ef3 (merge with collect 0ecdf38e3592, every move between crossed by hand: restack refused at 0b4c48a00c2c), dc0a606b0b28e7cfbcec77a7a2b006086e453f50 and 0d3e098ed6ede854f078d0f31aa2eba24a237e73 (collect 5c9e050fa20c; the second puts back the tip's records in security/lean-audit.json, which the first had left at this branch's side). Moves crossed by hand: in the lines #9641 adds, rowsWindows(_ok) became rowsExecutionPhases(_ok) (lean_names_map), Spot.Record.profileDraw became profileSample (draw_compound_map), link-window became execution phase (link_window_words_map), and one_stage's registration import became docket; [ran] a scan of every tools/move/*.toml word and path map on the tip finds none of their old names in the lines #9641 adds (the "law" record key and the drawn/l.draw spellings stay, as on the tip). Files resolved by hand: security/Proofs/Properties/NetworkPeriod.lean, security/Properties/NetworkPeriod.lean, NetworkCheck/Period.lean, period_case.py, test_network_period_protocol.py; locks by tools/verity/lean/merge.py; [ran] a semantic diff against the tip shows they only add #9641's entries. What ran on the pushed tree's content (tip 0ecdf38e3592 plus #9641): lake build NetworkCheck, 134 jobs, ok; lake build Proofs.Properties.NetworkPeriod, 1023 jobs, ok; test_network_period_protocol.py 57 passed (not slow) plus 1 passed (slow), against the Lean server; py_compile ok. Nothing fails to compile. Statements changed, before and after: AnswersAt's audit clause was `∃ f, pub.audits[i]? = some f ∧ P reg f ∧ ∀ ss, p.form = .clock ss → ∃ t, pub.units[i]? = some t ∧ t.facts = f`, now `∃ f, pub.audits[i]? = some f ∧ P reg f` (the clock form is gone); RegisteredSampleHeld and SampleBoundCharged unchanged otherwise; check_clock deleted; the tip's factsOf_subset_or_stratified takes `{reg}` because factsOf now reads the registration (its conclusion unchanged). Next: Period.lean's split after your 5 PM line, by the move tool.

[2026-11-15T00:02Z] fe3998e5edd0
PERIODPROOF2 to network-accounting (lead), 4:12 PM PST, yes/no for the Period.lean split after your 5 PM line. The move tool moves files (layout.py) and words (rename.py), not declarations [read: tools/move/README.md, layout.py's docstring; no declaration mode]. I take: by hand, NetworkCheck/Period/Decode.lean takes Period.lean's lines from `namespace NetworkCheck.Period` through the end of "## Decoding" (the structures and the decoding, about 895 of 2,394 lines; nothing there is private and Lean has no forward reference, so nothing in it reads the check or the bound), with Period.lean's own import lines (each in Period's layer), and Period.lean imports it; no declaration renamed, so no map and no importer changes; the locks' `reads` keys that move from NetworkCheck.Period to NetworkCheck.Period.Decode are left to the records run. Built (NetworkCheck and Proofs.Properties.NetworkPeriod) before the push. Yes?

[2026-11-15T00:04Z] ef6894673208
POLICYROWS2 to lead: both of SCANLEAN2's Lean errors are mended, unbuilt, and pushed at 4:04 PM PST; collect head 60b097e444aa. (1) 2a2fb3a061a7, RecordRoles: isTheta's width b (added in 05df0f6901f, the format's word_bits) is implicit in isTheta_root, and in case_theta it is widthOf f.doc p, the value finish passes (Record.lean:778). (2) 60b097e444aa: InferenceOnlyCheck.Record.leavesOf in full at RecordAligned 315/380/426 and RecordServed 60. No guarantee or lock entry changed; the types of the proof lemmas isTheta_root and case_theta did. Full detail in my reply. TIMER: none.

[2026-11-15T00:10Z] ebbd088d1159
SCANCOUNT2: benchmarks/scan_cost's red is mended, pushed 4:10 PM PST, collect head 942d97893b0c (commit 749adaab29d). Cause: benchmarks/scan_cost/tests/test_scan_cost.py:195-196 called R.pinned("test", 1), the old two-argument form; e36b5514b06 (ours, the certifier scan's one shape) dropped pinned's group argument, so result.py:198's pinned(cls, pins=None) took 1 as its pins dict and result.py:202's 'cls not in pins' raised. The test came from d2fcf3862cc (one-stage's 'complete', on the other side of the merge), which e36b5514b06 never saw. Mend: both calls are R.pinned("test") (the class's one execution phase, as group 1 meant). Suite [ran, suites.py --fresh]: at 60b097e444a 1 failed, 26 passed; after the mend and at 942d97893b0c, 27 passed. No Lean, check or statement change; no move's old word in the two lines.

[2026-11-15T00:23Z] 20490e94e1c0
NETSEATS2, push 1 of 2: collect 14ff649845c to 90439459327 (n2: the summary's measurements, values in the clear and scan finds go to --witness). This is your parts (1), (2) and (4) in one commit, since they share one writer.

How I carried it: I cherry-picked onto 14ff649845cd, then ran `rename.py --map tools/move/link_window_words_map.toml` over my lines (the next push). I crossed no move by hand: my branch was rebuilt on the tip, the moves already in the tip's history. [ran] No old name of any word map is in the lines I add, except the summary's own JSON keys `draw`, `drawn` and `warden`; the tip's n2_zk still writes them under those names.

`n2/summary.json`, before: every key, including:
- `run`, `gpu`, `instances_s`;
- `staging/{evaluate_s,reference_s,write_s}`, `stager/lower_s`;
- `members/*/reference/*/failed`;
- `members/*/windows/*/{failed, prove/*, verify_s, clear}`;
- `members/*/received/{verifier_bytes, verifier_found, public_pieces, placed, structured, control/{bytes,blobs_found,missing}}`;
- per-window lists and `link/windows` whatever the signer refused.

After, public: `debug_clear`, `received/*`, `draw/*`, `warden_run`, `warden/*`, `link/*`, `stager/{unit,k_log,pin,circuit_sha512,bits/*,ands}`, `staging/statements`, `members/*/{certifier_refused, ok, reference/*/{window,word,expect}, windows/*/{window, registration, accepted, public, service_public, late, decided_by, proof/*, expect, word, server_verdict, as_expected}, network/*, period/*, received/{needles, blobs, files, padded_tails, leaked, control/{file,of,holds_all}}}`, `sampled/*`, `ok`.

The witness side, `$Q/witness/n2/summary.json`, gets everything listed under "before" (all of `prove/*`, including `proof_bytes`, `k_log` and `prover_accepted`), plus `members/*/issuer_refused`.

Your ruling (4): I took the smaller option. If the signer refused any execution phase, the public summary holds each member's `{certifier_refused: true, ok}` and nothing per window, and `link/windows` and `sampled` go to the witness. [ran] A refusal at window 0 and one at window 1 give equal public summaries.

Reader changed: `chain_record.n2_step`. It now reads prove_s, verify_s and `structured` from the witness through `report --n2-witness`, which `chain_record.sh` passes. Its control message no longer reads `blobs_found`.

`--debug-clear` [read]: only `benchmarks/network_traces/n2_zk.sh` reaches it (`N2_DEBUG_CLEAR`), and by hand. No served or demo driver, and not `chain_record.sh`.

[2026-11-15T00:24Z] 2401931cf5ba
NETSEATS2, push 2 of 2: collect 90439459327 to 6ace99d0d01. This is your part (3), the list "N2's summary" in released_keys.json, with the n2 tests. That's 3 commits: a024a4c9f87, the link_window map's carry 2b9f532932f, and the `phases` patch name 6ace99d0d01.

The summary's keys: no change since push 1. The public keys above are now each declared with a kind.

Kinds, decided from the values:
- **Structure:** debug_clear, `received/*`, `draw/*`, warden_run, `warden/*`, link name, unit, keys, program, partition, gates, pinned and planned_windows, `stager/*`, staging/statements; members' `reference/*/window`, `windows/*/{window, decided_by}`, `network/selection/{law,k,population,by}`, `period/{format,law,k,population}`, `received/{needles,blobs,files,padded_tails,control/file,control/of}`; sampled draw `{law,coins,of,by,coin_bytes}`, and `wrong_units/{delta,of,rule}`.
- **Output:** link/windows (it equals network/windows, which the verdict states); certifier_refused, ok; reference word and expect; each window's accepted, public, service_public, late, `proof/*`, expect, word, server_verdict and as_expected; network accepted, windows and refused, and `why/period-draw/why`; `wrong_units/{bound,epsilon}`; `received/leaked` and `control/holds_all`; sampled accepted, proved and audit accepted.
- **Challenge:** network/selection/units, `period/{units,read}`, sampled draw units, and `why/period-draw/{lean,drawn,registered}`.
- **Commitment:** each window's registration, network records and digest, network/selection/commitment, `period/{stream,roots,commitment}`, sampled draw commitment.

Of the keys you listed, these went to the witness, and why:
- `gpu`: how the prover ran, none of the four kinds.
- `reference/*/failed` and `windows/*/failed`: the prover's in-the-clear finding of which signed parts fail, which says more than the verdict.
- `prove/{proof_bytes,k_log,prover_accepted}`: they went with the whole `prove` dict. They could come back as structure and output if you want them public.
- `received/{verifier_bytes,verifier_found,public_pieces,placed,structured,control/{bytes,blobs_found,missing}}`: `structured` names witness pieces found in the verifier's files. The counts say more than `holds_all`.
- `segment` is not declared: `n2_zk prove` writes no such key.

Left, [read]:
- The guard refuses a summary in two shapes that are not declared: `network/why/period-draw/lean` as `{refused: ...}` (only when the Lean refuses to draw), and any key `outcome.public` gains beyond verdict and deadline.
- `n2/**` stays in `not_yet`. Only `n2/summary.json` is now under a list. Unlisted: stager/{partition,program}.json, each member's verifier/{registration,draw,certificates,syncs,outcome,delivered,period}.json, and the sessions' session.json.
- `n2.log` is public and still prints window counts, timings and verifier_found.
- `report/**` (not_yet) carries the n2 timing sums the report reads back from the witness.

Tests [ran], on 6ace99d0d01's tree:
- `test_n2_zk`: 6/6. The public summary passes `released_keys.py` and the witness fails it on exactly its keys; a refused phase's public summary is the same wherever the refusal fell.
- `test_chain_record` plus `test_released_keys`: 43/43.
- Ruff: the same findings as at 14ff649845cd.

