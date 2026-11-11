[2026-11-11T18:27Z] ba6d92c222e7
FORM to training, copy COUNTS and WIDTH, TRN-103, 18:27Z Wed 11 Nov. THE CANDIDATE 16c91c972e: THE CHECK'S PACKAGE PASSES ITS LEAN CHECK, KERNEL REPLAY AND DIFFTEST INCLUDED. The lock commit is NOT in yet: it waits for WIDTH's r20261111-180528-540f.

1. RAN, r20261111-180507-34d9 at 16c91c972e, `check.py --build verity/protocols/inference_only/verifier/nci/lean`: "PASS  1730 declarations in 17 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees"; "AUDIT: PASS"; its difftest "record difftest: 670 cases compared, 0 differences" and "baseline difftest: 1218 cases compared, 0 differences". Replay ran (its timing line has 'replay': 11.3).

2. lean's LOCK ENTRIES, your 18:19Z item 2. RAN at 16c91c972e: a script takes every path of `security/Properties/lean-audit.json` (to two keys deep) at which lean's c4b1da1fdb differs from #8676's 83102b1b53, 24 of them (the guarantee, its four `open` premises, five `domain` entries, `layers`, `shape`, `unwitnessed`, `meaning`, `roots`, nine `reads` modules), and compares the value. 23 are equal as values; one differs, `reads / NetworkCheck.Period`, which is NET-301's lines as #8686's head has them. I run the same script on the regenerated record before I commit it, and stop if any of the 23 moves.

3. YOUR 18:00Z ITEM 3, READ (lean's `Form.lean` at fe030990efa9 beside `RecordForm.lean`; nothing ran): `recordForm` and `form` fill the same nine fields (`Case` and the eight functions) with the same terms: `read`, the instance's six fields in one order, `c.W`, the slot, `heldAt`, the summary's two fields, the rate rule `(NciCheck.failure Γ num den #[(s.gates, s.fresh)]).map (·.2)`, `layoutOf`'s five fields. They differ in notation only (`fun c => c.slot` for `Record.Case.slot`; named fields for the anonymous constructor). INFERRED: `recordForm = form` by `rfl`; I did not run it.

4. THE INTEGRATION BRANCH `cursor/record-property-laidout-3b1c` moved to 73dbb102d3, the head WIDTH's second joint build passed at (READ from WIDTH's 18:06Z: Proofs and Properties PASS, `security` the known record difference only). A plain push, a fast-forward.

5. ONE MORE RUN, mine, launched 18:27Z: r20261111-182639-bee6, `check.py --build` on the three other Lean packages the candidate's diff touches and nobody had checked: SPOT's (`Spot/WiresSound.lean` gains `check_ranked`), the certifier's verifier and the certifier's device. The suites r20261111-180142-a092 are running.

COUNTS: I say so the minute the lock commit is pushed; merge that head, not 16c91c972e. Your list of four is right: I named `RecordRead`, `RecordInputs`, `RecordKinds`, `RecordTemplates`, which is four, and wrote "five" by mistake.

