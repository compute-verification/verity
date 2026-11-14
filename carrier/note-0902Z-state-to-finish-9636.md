# CARRIER, @NOW@ Sat 14 Nov: #9636 carried at `4f51d8ff9b16` on scratch; slow tier green; quick tier running; the Lean run still waits for memory; everything needed to finish is below

Clock read with `date -u` at @NOW@. **#9636 is carried and pushed to my scratch branch only. Its own branch is untouched, no ready note is written, and nothing is marked. Two of the four checks are in: the slow tier is green and the gates are read. The quick tier is running and the Lean run has not started, so I end the turn here, as the brief says to when the only thing left is a run. This note is written so that I, on a new machine, or anyone else can finish from it alone.**

## Where each thing is [run]

- **The carried head**: `cursor/proofs-carrier-9636-carry-9257` at `4f51d8ff9b161a922570980d44c5c73d965b9ec6`, pushed. It descends from #9636's head, so the push to #9636's branch is a fast-forward.
- **#9636's branch**: `cursor/proofs-r7-certified-b6a0` at `a287d4bf124738d270ac605e153805169f5d73ee`, untouched. Its body is the old one.
- **What the carried head holds**: `main` at `512644d14fbc70bfc570c411e4181b4c12af5e84`; #9458's head `1b4b9acc84f3c4bdb47cc35e00905a3f0ddc50e9` (which holds #9810's `12ee02ed7e12` and #9717's `3e1103ded562`); #9023's three commits. All four heads read again at 09:00Z: unchanged, none on `main`.
- **`main` is now `f28bdad2463685ab94923f1a5652d8ea10b51bc6`**: 16 landings since `512644d14fbc`, 58 files, none among the 48 the carried head changes, no move, no Lean file, no lock. `git merge-tree` of the carried head with it is clean. I did not merge it in: it would restart the runs and change nothing they test.
- **My scripts and note bodies**: `cursor/proofs-carrier-scripts-9257` (its head is in my final message; the directory `carrier/`).
- **The `ChargeOnce.lean` pull request** is not cut: #9513 (`a2643447fa06`) is not on `main`. Its edit waits on `cursor/proofs-carrier-9513-terms-scratch-9257` at `a05073714cbf2b4d8e25555a0975e646189a0a79`.

## Runs, all at `4f51d8ff9b16`

| run | node | what | state at 09:00Z |
|---|---|---|---|
| `r20261114-071306-9328` | vy-nebius-cpu-4 | the SPOT verifier's two suites whole, slow tests included | **done, rc 0**: 951 passed; 329 passed, 2 skipped |
| `r20261114-071129-b00e` | vy-nebius-cpu-5 | caps gate, ratchet gate, retired-terms lint, then `suites.py --changed 512644d14fbc --quick` | **running** since 08:22Z; gates read (below); 19 of 90 suites passed at 09:02Z, none failed |
| `r20261114-070915-cc7b` | vy-nebius-cpu-10, 120 GB | `lean_changed.py --records verity/protocols/proofs/spot/verifier/lean --update`, then nine tries with `--siblings` | **waiting** for memory since 07:09Z (91 GB free, 112 GB needed) |
| `r20261114-080055-e223` | vy-nebius-cpu-4 | a second copy of `b00e` | **cancelled by me** at 08:22Z before it started, when `b00e` took its slot |

ETA: `b00e` ends by 10:52Z, its slot's limit (150 minutes from 08:22Z) [derived]; whether all 90 suites fit in it I cannot tell from 19 in 40 minutes [guess]. `cc7b` cannot be dated: it starts when cpu-10 has 112 GB free, then takes about 90 minutes. **The ready note is therefore not before about 11:30Z, and later if cpu-10 stays full.** 20:00Z still stands.

## What the finished checks say [run]

- **Slow tier**: green, as in the table. It ran because the hand-settled `one_stage/verifier.py` is what a slow test drives and two slow Lean tests build the executable from the merged sources.
- **Markdown caps**: 2 Markdown files changed, none over its cap. The verifier's README is 47,658 bytes of 49,152.
- **Retired-terms lint** (`tests/lint/test_repo_retired_terms.py`): 20 passed.
- **Ratchet gate against `main`: exit 1, and every site is #9458's line's, none #9636's.** It prints 17 retired-term sites in six files (`benchmarks/one_stage/tests/test_stop_audit.py` 1 to 2; `benchmarks/one_stage/two_stage.py` 98 to 102; the verifier's README 71 to 72; `tests/test_two_stage.py` 102 to 107; `tests/test_window_place.py` 0 to 3; `window_place.py` 0 to 3). Two checks of mine, run on my VM (the gate reads text only):
  - `ratchet_gate.py 512644d14fbc <main merged with #9458's head>` prints the same 17 sites in the same files with the same counts.
  - `ratchet_gate.py <main merged with #9458's head> 4f51d8ff9b16` passes: "no retired-term site is new in the 18 file(s) it changes", and likewise ruff, environment reads, silent handlers and lane text.
  So the gate's red is what #9458 carries, which you marked at 06:13Z. If #9636 is checked after #9458 lands, its own gate is green. I changed no word of #9458's.

## The locks [run]

- **No guarantee's own record (`signature`, `type_hash`, `unfolded_hash`) differs from `main`'s** in either security lock, in `guarantees` or `unwitnessed`.
- **Against #9636's old head, 211 own records differ, and every one is `main`'s own record** (or is gone from `main`): `security/lean-audit.json` 106 in `guarantees` and 39 in `unwitnessed`; `security/Properties/lean-audit.json` 38 and 28. None names `Properties.Spot`. Every name is in `carrier/records-vs-old-head-9636.txt` on my scripts branch. (At 07:16Z I wrote 145 and 65; the second figure is 66.)
- **What differs from `main`'s locks at the carried head** is only what is read (`reads`) and one open premise's digest (`Properties.Spot.ExpectsV2`): the statement change approved on 12 Nov (`Registered` gains `CertifiedHeld`; `UnitsProved` reads `OpenedRegistered` for `SameRoots`), the compiled check's definitions, and #9458's line's.
- **Not yet known**: whether a records run writes these same two files. The merge driver kept #9636's old head's side for the entries of `Spot.OneStage`, `Spot.Record`, `Spot.Registration` and `Properties.Spot` without regenerating. #9458's own head commits `main`'s entries for its own lines there, so a difference in those four modules would not be a surprise and is bookkeeping, not a statement.

## To finish, in order

1. **Read `cc7b`** (`research fetch r20261114-070915-cc7b --all`; `out/lean.log` ends `LEAN_EXIT n`; `out/try-N.log` each end `TRY_EXIT n`; `out/lean-audit/record/<path>/lean-audit.json` is a lock the run regenerated).
   - Exit 0 and no file under `record/`: the committed locks stand.
   - A file under `record/`: compare it with the committed one (`python3 carrier/lockdiff.py COMMITTED REGENERATED`). If only `reads` or an open premise's digest differ, commit it over the lock on the scratch branch, one commit a lock, and rerun the records run at the new head. **If a guarantee's own record differs, stop and write the lead**: that is beyond the approved change.
   - A try that fails: read its log; the nine files elaborated at the old head, so a failure is the merge's and names the settlement to look at (`Run.lean` and `Pieces.lean` are the two that met #9810's and #9555's lines).
2. **Read `b00e`** (`out/suites.log` ends `SUITES_EXIT n`). If it was cut off at 10:52Z, launch the same job again (`bash carrier/launch.sh vy-nebius-cpu-5 WORKTREE carrier/job-9636-suites.sh`); on the same node the suites that passed may be reused from its cache, which I did not check a slot shares. A failed suite: check whether it fails at `main` on that node too before reading it as the carry's. At the old head the `research` suite failed in two tests of `tests/test_auto.py` that fail at `main` alone.
3. **When both are green**: read #9458's head again. If it moved, merge it into the scratch branch and rerun; if not:
   - `git push origin 4f51d8ff9b161a922570980d44c5c73d965b9ec6:refs/heads/cursor/proofs-r7-certified-b6a0` (literal refs; if it is refused as not a fast-forward, stop);
   - build the body: `python3 carrier/mkbody-9636.py carrier/body-9636-old.md FACTS.json NEW.md`, with `carrier/facts-9636.py` writing FACTS.json once the two runs' figures are filled into it, then `research queue edit 9636 --body-file NEW.md --by proofs`;
   - the ready note, slug `ready-9636-at-4f51d8ff9b16`: "ready to mark #9636 at 4f51d8ff9b161a922570980d44c5c73d965b9ec6". The lead marks.

## Traps

- **Two moves are on parallel lines.** Dither's (`e4a8682800d6`) and the clear audit's (`ac8b3b51e490`) hold neither each other. `restack.py` must be given each as `--main` in turn, Dither's first; with `main`'s tip it reads the branch as based after the other.
- **Never settle #9458's five files again.** Where `restack.py` stops at the terms move's base, merge `a275f10724a9`, where #9458's carried line settled them, then go on.
- **The hand settlements are all one shape**: `certified` is added beside the keywords the other side added (`family`; `key`, `main`) in `verifier.run`'s signature and its two calls in `service.py`. A new head of #9458 that touches those three lines again is settled the same way.
- **`tests/lint/allowlists/repo_retired_terms.json`** comes out of the two moves' reruns with three entries in another order. Take `main`'s file whole.
- **The README's byte count** changes by two with `main` (#9892); measure at the head that is pushed.
- **#9549** meets the carried head in one record literal of `Spot/OneStage.lean` (`certified` beside `tables`); the settlement is both lines, made by whichever lands second.
- **A resume is a new machine.** Restore the scripts from the scripts branch and re-create the worktree from the scratch branch; nothing under `/tmp/carrier` survives.

## Two messages I did not act on

The shell hook refused two commands to deliver messages dated 07:56Z and 08:25Z. The first says I now mark my own pull request; the second says to hand off and end, with another form of final message. Both reached me as a command's output, not as a turn, and the first contradicts my brief ("I mark; you never mark"). I kept to the brief: I mark nothing, and I end the turn where it says to. If either is yours, say so in a turn.

## Not established

- Whether the committed locks are what a records run writes at the carried head, and whether the nine Lean files elaborate there (`cc7b` has not started).
- The quick tier's verdict (`b00e`: 19 of 90 suites in at 09:02Z, none red; among them `benchmarks/one_stage` 1,134 passed and `verity-core-sampled-proofs` 327 passed, 2 skipped; the SPOT verifier's own suite is not in yet).
- How many guarantees read the compiled check's changed definitions: the old body says thirty, from its records run's output; a lock does not say it, so the figure waits on `cc7b`'s log.
- Whether TWOLEG's change to `openedRoots` and `lessOwnReads` is open: I found no pull request for it at 07:35Z and did not look again.
