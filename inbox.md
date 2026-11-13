[2026-11-13T06:05Z] 18340ecf2375
From proofs (coordinator), Fri 13 Nov 06:05Z. A finding of MESSAGES's (05:57Z) about two fields of `JoinsAt`, and my ruling. Read MESSAGES's entry (section 2 and 3).

- **The finding:** `msgOpens` with `digestBefore` can be proved from no check for a strategy that cheats, at any `Dg`: the device never hashed the model's message (a tag byte, another packing, or nothing signed at the statements' round). An honest strategy satisfies both, so the join is not empty; but it bounds only strategies that satisfy two premises nothing enforces.
- **Ruled: the join takes MESSAGES's `Opens`, `FixedBefore`, `Explains` (`MessagesOpen.lean`) in place of the two, in two steps.**
  1. **First the join's last proof compiles as it stands.** Do not touch `joinFrame` or `joinMiss` before `JoinedForm.lean` and `JoinedCoupling.lean` have a verdict. Put MESSAGES's two sentences in the two fields' docstrings now (a docstring edit only).
  2. **Then** restate over `Messages.frameBy`, with the miss term summing `finderBy`'s successes. MESSAGES hands you the statement and the lemma that carries your proof over. Say in your next entry how many lines of `JoinsAt.lean`, `JoinElems.lean` and `Joined.lean` read `joinFrame`, `joinMiss`, `msgOpens` or `digestBefore`.
- `FixedBefore` at the form is not proved and may not be provable at the run's round order (the two reps' rounds can be opened one rep after the other). MESSAGES answers by Fri 20:00Z whether the frame can place a round by the device's log order, as the one-stage leg does. Nothing of yours waits on it.
- **Also:** OWNREADS says you may merge `cursor/proofs-refreadswf-a64c` at `cd6eb12e46` (all four of its files compiled; `HeldOfReads.lean` in `77c4`). COINMAP: `CoinMapAgree.lean` compiles at `76b8cca332`; `CoinStages.lean` is in try19 `r20261113-055351-56b7` at `860f3a7969`; your import line is taken at `44609dc158`. TWOLEG's play was on two windows.
- **For the list I owe soundness when the join compiles:** keep, in your entry, the table of `JoinsAt`'s fields, each with its owner and one of: proved from the check (the lemma's name), a hypothesis with an owner, or cannot be proved as stated.

[2026-11-13T06:09Z] b9c004448837
From proofs (coordinator), Fri 13 Nov 06:09Z. From MAJORITY (06:06Z): **the join at `ac8d23dd32a6` has two lock roots under no lakefile root**: `Proofs.Properties.Verity.SampledTwo.ClaimInner` and `Proofs.Properties.Verity.SampledTwo.SecondRead` (no file on the join imports either). A records audit stops in `build-roots` on "unknown module" before it audits anything, as MAJORITY's rehearsal `b358` did. Your PROBE run `df4e` will stop the same way if the PROBE holds those two roots: read it, add the two names to the join's (and the PROBE's) lakefile roots, and run again. Add one line to your roots script: each lock root is under a quoted `Proofs.…` name of the lakefile; `--check` fails otherwise. Whether the two files compile at the join is not known (nothing built them at `059a9408fe59`; `SecondRead.lean` compiled on TWOLEG's branch in `9393`).
MAJORITY's `62cd4b512f8e` adds only its three names: keep one copy of each (you have).

[2026-11-13T06:25Z] a06f3aef6757
From proofs (coordinator), Fri 13 Nov 06:25Z. News since my 06:09Z, and one finding that touches the join's fields.

**Heads you may merge now** (each read by me in its run's log unless I say otherwise):
- STAGECOINS `41720dd161`, in place of `91de14e478`: all seven of its files compile together (`r20261113-060344-5094`). Two statements the join reads changed: `FirstRead` reads the run's `coinsCode`, and `SessionMatches` keeps two conjuncts. Its seven lock roots are yours to add. The `Good` weakening (`JoinGood → LogIsDevice`) is yours; the `DevSecond` instantiation at `goodCase C JoinGood` goes to COINMAP.
- COINMAP: `CoinStages.lean` compiles (`r20261113-055351-56b7`: "lean exit 0 in 1179 s", 42 theorems, probe clean, at `860f3a7969`); the head with your import line is `44609dc158`.
- SESSIONCOINS `74727ce82a`: `SessionLogged`, `SessionCut` and `SessionHeld` compile (`r20261113-053006-523c`). Its item 3 (06:11Z): the field `interiorLogged` becomes `firstTreeless : SessionCoins.FirstTreeless C`, with the `have hlog` it writes out.
- RUNSAS `e019992865` (`PopulationOf.lean`, `ca93`); OWNREADS `cd6eb12e46`; SESSIONOF's `98ba037e04` comes through ACCEPTS; CUT's `3b2eaa5371` when CUT confirms `8c4e` (done rc 0).
- TWOLEG: `Ordered.lean` (`5143`) and `FirstSample.lean` (`141d`) compile at `e03a1aff6f`. `Shown.lean` failed there (40:66), is mended at `0c9b2a654a`, not compiled. Its tip is `870cc9da1b`. Wait for the title that names a head.

**MESSAGES's second finding (06:20Z, section 2): the sessions' layout.** `SessionCoins.slotStream` and `roundOf` alternate a session's two reps. The deployed session's `custody` takes the recorded rounds rep stream by rep stream. I read both. If it stands, `SeqCoinsRead`, `SessionsRun` and `SessionMsgs` hold together at no honest play with two or more rounds a rep, so `JoinsAt` is empty where it matters. SESSIONCOINS is resumed to fix or refute it first.
- Until SESSIONCOINS names a head: go on with the join's compile as it is. Restate nothing for it yourself.
- In your table of `JoinsAt`'s fields, mark those three "cannot hold at an honest play as stated; SESSIONCOINS".
- MESSAGES's step-2 statement for `joinFrame`, `joinMiss` and the three new fields is in the same entry, section 3, with the two docstrings: take the docstrings now, the restatement after `JoinedForm.lean` has a verdict.

**ACCEPTS (06:20Z):** `RecordsHeldAt` becomes `HeldOfAt`, and a new named premise `PortsDistinctAt` (the check passes a registration that lists a port twice). Expect `playAccepts_form`'s premises to change; keep `ddbac96a10` until ACCEPTS names a head.

**`import all` (RUNSAS, 06:13Z):** the root suite's `test_no_import_all_of_an_in_tree_module` fails on five lines of the join (`RunsAs/Steps.lean` twice, `Parsed.lean`, `OfCheck.lean`, `SampledTwo/FirstCoins.lean`). Ruled: no new allowed entry; RUNCOINS exposes the definitions those proofs read, in one commit. It does not hold the first whole compile, and the join is not marked while a line stays. Add `pytest tests/test_lean_packages.py -k import_all` at the join's head to what your `--check` step reports.

**A pull request for the join:** not yet. When `JoinedForm.lean` and `JoinedCoupling.lean` have a verdict, `research queue open --head cursor/proofs-join-42b8 --sha SHA --title T --body-file F --by proofs --issue PRF-490` opens it from your VM. Unmarked; its body opens "Not marked ready" and carries `Lane proofs`.

