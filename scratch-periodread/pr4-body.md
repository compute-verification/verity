Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. The fourth of its pull requests: the proofs of the five statements of #9581, on its head. It changes no statement and adds no guarantee entry.

**What "proved" means here.** Each of the five theorems elaborates and rests on `propext`, `Classical.choice` and `Quot.sound` only. Two checks are not mine to give: the kernel replay, which only a candidate's check runs, and a second reader's verdict on the statements, which is not in.

## The proofs

- **`security/Proofs/Properties/Spot/Period.lean`** (815 lines): `ledgerMeans : LedgerMeans`, `acceptMeans : AcceptMeans`, `committedOnce : CommittedOnce`. No premise.
  - The check is monadic. What each of its functions read where it returns is a Hoare triple proved by `mvcgen` and read back at an `.ok` result; the properties are then facts about arrays.
  - **The ledger's half.** `read` keeps, of the windows the ledger's check returns, those whose open is a v2 entry of the lane in the monitoring period's run, so a session is returned exactly for a root the lane committed. `windowsIn` splits them by whether their layout holds the sample's family. The checked ledger's windows share no root and no session, so neither do the windows and the others together.
  - **The audits' half.** `windowAt` either leaves an audit undecided or adds one window with one audit and its record. Along the fold over the windows, while no audit is undecided, window `i` passed with one audit, made at the records of the windows before it.
  - **Each committed root once**: the two together, a window's place being its root's alone.
- **`security/Proofs/Properties/Challenger/Lines.lean`** (190 lines): each line of a ledger the ledger's check accepts is numbered by its place, and says what a line of the device's own ledger says, or the finder outputs a break. Two entries whose texts without their `sig` agree have one number.
- **`security/Proofs/Properties/Spot/PeriodLedger.lean`** (91 lines): `ledgerLinesOfDevice : LedgerLinesOfDevice`, with no premise, and `ledgerOfDevice : LedgerOfDevice`, from `Numbered`.

## One change to a proof already there

`security/Proofs/Properties/Challenger/Window.lean`: the per-line step of `window_opens` is its own lemma (`line_of_device`), with `sig_lms` beside it, so the ledger's lines read it too. `window_opens` keeps its statement. Which records' `reads` regenerate: RECORDS

## `Numbered`

`ledgerOfDevice` takes `Numbered dev pk` as #9581 states it: a hypothesis of that statement alone, not a named assumption, read by no lock entry. **"The ledger is whole" is proved only from it.** Without it, a device that wrote two lines under one number satisfies `LedgerLinesOfDevice` while the presented ledger holds only one of them. The device does number its lines (`verity/protocols/spot/device/challenger.py:1503`, `"seq": len(lines)`); its model does not hold that yet.

## The lock

- **`security/Proofs/lean-audit.json`**: `roots` gain the three proof modules; the lakefile's too.
- **`security/Properties/lean-audit.json`**: each statement's `unstated:` line now names the theorem that proves it. `LedgerOfDevice`'s says it is proved from `Numbered` and gets no guarantee entry while it takes it.
- **No guarantee entry.** A guarantee entry for a property of an acceptance needs an honest witness: a ledger whose LMS signatures the kernel verifies, which the check accepts. None is written. So no ledger, table or page may cite these five as proved guarantees yet.

## What is not established

- **Nothing ties `CommittedOnce` to the pooled integrity statements** (`Flock.Guarantees.IntegrityPeriod` and the two beside it). #9581 says so in `IntegrityPeriod`'s docstring.
- **What one-stage v2's accept gives.** `acceptMeans` holds for any audit of one registration.
- **No accept is tested.** No proof in the tests is one that one-stage v2 accepts, so no ledger is known that the check accepts.
- **The kernel replay and the second reader**, as above.

## Checks

CHECKS
