Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. The first of its pull requests after #9416: the ledger's half of a monitoring period's check. It reads the auditor's ledger and says which sessions a lane opened on a root during a monitoring period. It audits no window, states no guarantee and adds no lock entry; nothing calls it yet but its difftest.

## What it adds

`Spot/Period.lean`, in SPOT's Lean verifier, compiled, with no Python verifier beside it. The verifier holds the auditor's challenger key, the lane, the family a sample's challenges are of (`label`) and the monitoring period's two ends.

- **`sessionsOf`**: every session of the lane that the ledger opens on a root during the monitoring period, whatever its layout. Each comes with its root, its families, whether it withholds its commitment to the challenges, its place among the lane's reserves and the ledger's period that lists it.
- **`windowsOf`**: of those, the sessions whose layout holds `label`, in the ledger's order. The lane's other sessions are returned beside them as `others`, with nothing held of them, so a reader sees every root the lane committed, a window or not.
- **The difftest's piece `period_windows`**, and `tests/test_period_difftest.py`, which holds the Lean to a Python reference that lives in the test alone.

## What it refuses

One code each, the first in this order:

| code | when |
|---|---|
| `coin` | the ledger does not verify under the auditor's key (`Challenger.checked`) |
| `early` | no entry is stamped at or after the monitoring period's stop: a ledger cut short could hide a later session |
| `missing` | the lane opens no session holding `label` in the monitoring period |
| `outside` | a window is in no period of the lane |
| `period` | a window's period is not closed, or lists an opened session that is no window |
| `missing` | a period lists a session reserved and never opened, or a window has not ended |

## The clock: a monitoring period is a run of the ledger's entries

This is my call, as the coordinator left it to this pull request, and it differs from the default he named (refuse a ledger whose stamps fall inside the monitoring period out of order).

- **Why not stamps alone.** The auditor's device that runs the challenger reads its clock before an entry takes the ledger's lock, so two entries written together can carry stamps out of order. An honest ledger then has stamps that do not rise.
- **What it does.** The monitoring period is the entries from the first stamped at or after `start` up to, and not including, the first stamped at or after `stop`. Monitoring periods that follow one another split the ledger's entries exactly, none in two and none between, whatever the stamps say. A ledger read again with entries appended gives each earlier monitoring period the same run.
- **What the default would cost.** Refusing a ledger whose stamps do not rise would refuse the honest ledgers above. The difftest has two of them: an end stamped before the entry written ahead of it, and a window stamped before `start` by a clock set back.
- **What neither gives.** That a session's stamp is the time it ran. The stamps are the auditor's device's, and the device is trusted (`trusted-host`).

## What it does not do

- **No window is audited.** That is the next pull request, which calls one-stage v2's check on each window.
- **No statement, no lock entry.** The statements come in their own pull request.
- **It does not ask a window's end to say `complete`.** The gate's `complete` means every challenge of every family was released. An audit reserves at least 16 challenges a rank and releases what its sample needs, so an honest audit's end says `complete: false`. I read this in the gate and the audit's layout of challenges; I looked at no served audit's end entry.
- **No executable.** The reader is served through the difftest's table only.

## What the difftest holds

- **Lean and the reference agree** on 60 sampled ledgers a small key signs, written one entry at a time as the gate writes them: a lane's periods of sampled windows, other lanes' sessions among them, the lane's sessions of other layouts between its periods.
- **Each named case gets its own code** over three seeds: a ledger cut before the stop, an edited signature, another family asked for, a window in no period, a period left open, a period holding a session of another layout or one opened before the start, a reserve never opened, a window never ended.
- **Three ledgers that must still read the same**: an end stamped before the entry ahead of it, a window stamped before the start by a clock set back, a ledger of one-step opens.
- **One ledger written by hand** with its expected answer written out, the only case whose expected value does not come from the reference.
- **Monitoring periods end to end split the sessions**: ledgers with stamps in no order, read at three periods one after another, give each root once.

The reference and the Lean share an author. No ledger the gate itself wrote was read.

## Checks

- **Build and difftest**: `r20261112-202226-a838`, exit 0, at `7639eb10259f`: `lake build Spot sampled-proofs-difftest` exit 0; `test_period_difftest.py` 12 passed in 132 s. The head after it changes the test only (60 sampled cases where it took 120, three seeds a case where it took six, two tests marked `slow`).
- **Quick tier on the head** `44b84839217d`: 61 of 61 suites passed, in two runs. `r20261112-203704-77ca` passed 53 inside the line's one-hour cap, none failed; `r20261112-215441-6715` ran the 8 it had not reached: 8 passed, 0 failed.
- **Records run on the head** `44b84839217d`: `r20261112-231708-ddb2`, exit 0, `lean_changed.py --records verity/protocols/spot/verifier/lean --update`. It audited SPOT's verifier, the certifier's verifier, `security` and `security/Properties`, each with `security/Proofs` where it requires it: `AUDIT: PASS` for each, axioms `propext`, `Classical.choice`, `Quot.sound` only, and `regenerated` empty in each report, so no lock record moved. It is a slot-path result (`lean-fast/v1`): no kernel replay and no tests, which the candidate's check runs.
- **The ratchet gate fails at this head**, on retired terms: 33 new sites in the files this change adds or edits. 10 are the name `Period` of its own module and namespace, which the Glossary's row for `*Period*` names counts; 16 are the ledger's family label `draw` and the test's constant for it; 3 are the ledger's field `cap`; 3 say "sampled proofs" where the Glossary says SPOT; one names the existing `MonitoringPeriod`. The gate was not run before this change was marked. It is off the merge gate and holds nothing. The vocabulary test warns on two structures, `Spot.Period.Session` and `Spot.Period.Window`.
- **Merge**: the head holds `main` at `5f0c008aa0`; against `main` at `e86d920413`, `git merge-tree` exit 0.
