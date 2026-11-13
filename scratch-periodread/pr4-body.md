Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. The fourth of its pull requests: the proofs of the eight statements about SPOT's period check. It is on the statements' head (#9581), which is on #9595's and #9537's; all three land first.

**Each of the eight statements elaborates on `propext`, `Classical.choice` and `Quot.sound` alone, and none gets a guarantee entry.** No full Lean check has replayed them through the kernel on this head, and no reader other than their author has read the statements as they stand.

## What it proves

`security/Proofs/Properties/Spot/Period.lean`, with no premise:

- **`ledgerMeans`**: what the ledger's half returns. Every root the lane committed in the monitoring period is returned once, a window when its session can be sampled and another session when it cannot.
- **`acceptMeans`**: what the check accepts. Each window has exactly one audit, the check's own at that window.
- **`committedOnce`: every root committed under this registration is one published window, with one audit entry.** Every root the lane committed is on a session that a period committing a circuit lists; under this registration's circuit it is a window, and its place is its root's alone.
- **`othersElsewhere`**: the published others are exactly the roots the lane committed under another circuit, each once.
- **`decodedMeans`**: the key each audit is handed reads as the key the ledger is checked under, and `audit` names none of the keys a window sets or leaves unset.
- **`auditedSession`**: the session audited at a window is the window's own.

`security/Proofs/Properties/Spot/PeriodLedger.lean`:

- **`ledgerLinesOfDevice`**, with no premise: each line the check reads says what a line of the device's own ledger says, or the finder outputs a break.
- **`ledgerOfDevice`**, from the one hypothesis `Numbered`: the presented lines are the device's own from its first line, none dropped and none added, or a break.

## How

- **The check is monadic.** What each of its functions read where it returns is a Hoare triple proved by `mvcgen` and read back at an `.ok` result (`read_ok`, `windowsIn_ok`, `registeredIn_ok`, `caseAt_ok`, `windowAt_raw`, `run_ok`). The properties are then facts about arrays.
- **The refusal** (`registeredIn_ok`): where it returns, every window is in a period that commits this registration's circuit and every other session of the lane is in a period that commits another. `committedOnce` and `othersElsewhere` are that with `ledgerMeans`.
- **The fold over the windows** keeps one invariant: while no audit is undecided, window `i` passed with one audit, made at the records of the windows before it.
- **The ledger's lines** are in a module of their own, `security/Proofs/Properties/Challenger/Lines.lean`: `checked` reads every line as an entry numbered by its place whose signature verifies, and a line whose signature verifies is the device's or a break. That last step was inside `Window.window_opens`; it is now the lemma `Window.line_of_device`, which `window_opens` calls. `window_opens`' statement is unchanged.

## The lock

- **`security/Proofs/lean-audit.json`** and the lakefile's roots gain the three modules.
- **`security/Properties/lean-audit.json`**: each of the eight `unstated` lines now names the theorem that proves it. `LedgerOfDevice`'s says it is proved from `Numbered`.
- **No guarantee entry**, as in the statements' pull request: an entry needs an honest witness, a ledger whose LMS signatures the kernel verifies and which the check accepts, and none exists yet.

## What it does not do

- **`Numbered` stays a hypothesis.** It goes when the model of the device carries the numbering.
- **Nothing is proved of one-stage v2's accept.** The statements hold for any audit of one registration.
- **Nothing ties this registration's circuit to what `audit` describes**, and nothing is held of a root committed under another circuit.
- **No witnesses.** That some period is accepted is shown by the difftest's stand-in audit only.

## Checks

- **The two proof files elaborate, on the three axioms only**, on vy-nebius-cpu-11 at `80f04b1def7f`: `r20261113-035213-962c` (`Spot/Period.lean`, `lean exit 0`, "none of its 63 theorems uses an axiom beyond propext, Classical.choice and Quot.sound") and `r20261113-035417-2795` (`Spot/PeriodLedger.lean`, the same of its 4). The head after them changes eight lock lines.
- **Their imports build there** (`2795`): `Proofs.Properties.Challenger.Window` and `Lines`, `Properties.Spot.Period`, `Properties.Spot.PeriodLedger`.
- **On the head, by hand**: `tests/test_lean_packages.py`, 34 passed.
- **Not run on the head**: the Lean check (the kernel's replay and the locks' rules), `Proofs.Properties.Challenger.Drawn`, which also imports `Window`, and the quick tier. No ready line until they are in.

Refs PRF-600
