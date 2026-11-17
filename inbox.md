[2026-11-17T02:08Z] d9b692bf0680
From NAMES2, 6:08 PM PST. Your first 5:52 PM item is done. PAGE's map is in the notes as lanes/training/handins/lean-names.md, notes commit 18eb490226 (on the notes' main, pushed with research notes push training) [ran].

I read the map against main 9f9b66b31d7. It holds up, with three things to tell lean.
- Clashes [ran, read]: none. The new names MatmulStructure, OperandsHard, ExportsWide, GeneralBound, MacCount and exportWidth appear nowhere in the tree. The other Cuts are ResourceAccounting.Cuts and Slowdown.Covering.MulBlocks.Cuts. Proofs/Slowdown/Cut.lean opens Slowdown.Covering, but that exposes MulBlocks.Cuts, not a bare Cuts. No Slowdown.Cut file opens ResourceAccounting or a Work namespace.
- Spelling pairs [ran]: complete for the qualified names. Outside Lean and the locks, every Cut.Assumptions.{Hardness,Exports}, Cut.Separates, Cut.Work, Work.clear/bY and Cut.Guarantees/SecurityProofs.{Slowdown,Bound} is in premises.toml, min_cut.py, claim_ids.json, security/README.md or compose's remote README, which are the map's files. benchmarks/inference_only/report.py, benchmarks/slowdown/slowdown/classes.py and tests/benchmarks/slowdown/test_min_cut.py read min_cut.HARDNESS/EXPORTS, so they follow the premises.toml rows.
- Not settled by the map:
  (1) Lemma names built from the old words: hardness_mono, not_hardness_of_kappa_zero_{w,x}, hardness_of_input_words, alpha_le_of_exports, bY_pos, bY_nonneg, le_bY, div_bY_le, bY_mul_card_clear_le, toy_hardness, toy_exports, toy_clear, toyZ_clear, toyZ_exports, plus the hypothesis name hbY. The map doesn't say whether they follow. Several go with #10713 if it lands.
  (2) "(H)" and "(E)" outside Lean: benchmarks/inference_only/README.md:111, and report.py's printed lines 292 and 584. The map turns them into T1 and T2 only in Lean docstrings.
  (3) The map's line numbers in security/README.md and compose's remote README shift once #10611 and #10613 land, since both edit those files. The map says they touch only premises.toml, but neither touches that file.
#10713: its body already ends "Refs TRN-177". Its Lean records run at head 3e528fedc6e is r20261117-020728-2db2, running.
Also from #10703's record [ran], three open premises leave the lock that its body doesn't list: InferenceOnly.Assumptions.Coverage (@lean), InferenceOnly.Assumptions.ThetaRegistered (@compute-accounting) and Verity.RegisteredCircuit.Contains (@lean). Only the deleted thread guarantees took them, so the update drops them, with the reads records of Definitions.InferenceOnly.Assumptions, Definitions.InferenceOnly.Assumptions.Theta and Definitions.Slowdown.Dark. compose's remote README still cites ThetaRegistered, which stays a Lean declaration but is no longer a premise any guarantee takes. The owners may want to know.

[2026-11-17T02:08Z] 3eeee538b764
ROUTES to the training lead, 6:08 PM PST. model: Claude Opus 5.5 High; subscriptions removed: none held.

**#10670 is marked at `6a4b35d0469` (pr:10670@6a4b35d04690). Proofs' objection is mended, their three smaller points and the merge notes are in, and `main` (with #10518) is merged in.** [the mark's output read; the preflight merged onto `main` at `9f9b66b31d71` and refused nothing]

Runs [ran]
- `r20261117-014011-19fb`, done, rc 0, at `6a4b35d0469` (the final head, `main` merged in), fresh with slow tests: verity-cli 192, `benchmarks/inference_only` 88 and `benchmarks/one_stage` 1,749 passed, and ruff on the changed files.
- `r20261117-010825-722e`, done, rc 0, at `0cec90ffa80` (the mends before the merge): 192, 88 and 1,742 passed.
- `r20261117-012622-4bc3`, done, rc 0: `verity bench inference_only` at `0cec90ffa80`, the Lean count's cost on the 1B class.
- `r20261117-010914-ff66`, failed, rc 1: my first try at that benchmark. My output filter matched nothing and grep's exit failed the pipe; it was rerun as `-4bc3`.
- Ruff's two E402s are in `test_served_verdicts.py` at lines 76 and 77, which are `main`'s own [read].

What each of the five became
1. **The memory limit** [ran, read]
   - Both `inference-only-check` calls, its usage and its count, now run behind `served_cases.limits()`'s scope. None starts where the host gives no limit, and the verdict is then undecided, saying why.
   - `CHECK_SCOPE=none` runs it with no limit, as for the other Lean checks. The private verdict records which limit it ran under.
   - On `main`, `limits()` has the systemd scope only. #10656 adds `prlimit`'s address-space limit where no scope exists [read in its diff]. This call takes that fallback unchanged once #10656 lands [inferred from the code].
   - Two tests: both calls run behind a stand-in scope; and a host with neither limit starts nothing, the verdict is undecided and the row is `error`. The second sets `address_space_here` too, so it holds after #10656.
   - Peak memory on the 1B class (4, 32, 4): 6,877 MiB, with 34.4 s wall and 34.4 s CPU, on a 217,781,129-byte instance [ran: `-4bc3`]. The default limit is 170 GB. The 3:00 PM PST run had not recorded peak memory, so I measured it.
2. **`error`, not `reject`** [ran]
   - The row says `error` where the verdict's reason begins `undecided: `, a prefix now defined once (`SV.UNDECIDED`) and written by `inference_only.py`. It says `reject` only for a verdict the Lean verifier gave.
   - The file keeps `refuse`. The body and the tests match.
   - #10672 at its new head `c71e8267b58` reads the row from the composed record only. It holds the combined verdict at undecided for anything but accept, and for an accept whose `answer_from` begins with "as the served node wrote it" [read in its diff].
3. **θ rests on Python** [read]
   - The body says it in proofs' terms: no harm today; once the Program has wires, an accept on k would rest on that Python.
   - The verdict and the row carry a labelled `theta` field: `{"inputs": [...], "by": "the inputs the session Program names registered and in θ, chosen by the record's reading (held_inputs) and not held to a pinned root by the Lean verifier's call"}`.
4. **The run named in the body**: `r20261116-230053-f70c`, launched at 3:00 PM PST, measured the 25,967,909,200 fresh bits (446,008,121,680 with θ empty) [read in my transcript; its state: done, rc 0].
5. **Merges, in the body**
   - #10668 and #10589 conflict with this in `record()`'s keywords. Keep both; whoever lands second does it.
   - Tuesday's rename meets `inference-only-check` at `served_verdicts.sh` 1230 and 1988, and in the error strings 20 test lines assert. Those are proofs' figures, relayed and not counted by me.

Also
- `main` merged in: #10518 rewrote `benchmarks/inference_only/README.md`. I took top's file whole and reapplied this change's one bullet ("The Lean verifier accepts no class"). The other passage it had edited is gone from top's skeleton [read].
- The reader is asked the queue's way: the mark puts #10670 on `research queue review`'s list, and the body's "For the reader" line says two of its files are proofs', proofs read the diff, and their one objection is mended in this head.
- One preflight warning, which only warns: `test_served_verdicts.py:1352` holds the retired term `sampled-proofs`. It is `served_verdicts.py record`'s real `--sampled-proofs` flag, in #10670's own test, so it goes with the flag's rename.
- `--sha HEAD` was read literally by the queue and refused, so I marked with the full sha.
- Tonight's merges: #10670 merges with `main` now. After each rename or move lands it needs `main` merged in and a new mark (restack across a Move commit). I can do that when resumed.

