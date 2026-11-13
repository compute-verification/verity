Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. The third of its pull requests: what an accept of SPOT's period check means and whose ledger it reads, as five statements, with the check's executable. It is on the audits' half's head (`cursor/proofs-prf600-period-audit-9c4e`), which is on #9537's; both land first.

**Nothing here is proved, and no statement gets a guarantee entry.** The proofs are the next pull request. Each statement is held in the lock as a draft (`unstated`), the way the three drafts of `Properties.Challenger` are.

## The statements

**`security/Properties/Spot/Period.lean`**: three statements about the check's functions as they run, for every input their decoders produce. They take no premise. Each is about the ledger the check was presented and says nothing of whose it is.

- **`LedgerMeans`**: what the ledger's half returns (`windowsOf`). The ledger verifies under the verifier's key and reaches the monitoring period's end. Every root the lane committed in the monitoring period is returned: a window when its session can be sampled, one of the other sessions when it cannot. Nothing else is returned, and no root and no session is returned twice. Each window's session has ended, in a closed period all of whose sessions are windows.
- **`AcceptMeans`**: what the check accepts (`checkWith`, at any audit of one registration; the check itself is at one-stage v2's). It publishes the ledger's half's windows in order and its other sessions. Each window has exactly one audit, and that audit is the check's own at the window (`AuditedAt`): the session is the one the ledger sealed, run from a commit the pins log records on main's first-parent line, with its coin commitment withheld; its log shows the sample's coins released after exactly one registration; and the audit accepted that registration at the parameters the check composes for the window, after the records of the windows before it.
- **`CommittedOnce`**: the two together, as a reader of the published answer needs them. Under an accept, every root the lane committed in the monitoring period on a session that can be sampled is the root of exactly one published window, which has exactly one audit; every other root it committed there is published among the other sessions; and each published window is such a root.

**`security/Properties/Spot/PeriodLedger.lean`**: two statements of whose ledger the check reads, against the challenger's model of the device (`Properties.Challenger`: `History`, `Break`).

- **`LedgerLinesOfDevice`** takes nothing of the device. Every line of a ledger the check reads is an entry under its place's number that says, its signature aside, what a line of the device's own ledger under that key says; or the fixed finder outputs a break (an LMS forgery or a SHA-512 collision).
- **`LedgerOfDevice`** takes one hypothesis, `Numbered`. Then the presented lines are, their signatures aside, the first lines of the device's own ledger, in order, none dropped and none added, up to a line stamped at or after the monitoring period's end; or a break.

## `Numbered`: the one hypothesis, and why the second form needs it

`Numbered dev pk` says the device numbers its ledger's lines: line `i` of its ledger under the key is an entry whose number is `i`.

- **Without it the ledger is not shown whole.** A device that wrote two lines under one number satisfies `LedgerLinesOfDevice` while the presented ledger holds only one of them.
- **The device does number its lines.** `verity/protocols/spot/device/challenger.py:1503` writes each line with `"seq": len(lines)`. The model of the device (`History`) does not hold that yet.
- **It is a hypothesis of `LedgerOfDevice` alone, not a named assumption.** No lock entry reads it. Its docstring says it goes when the device's definition carries the numbering: then it is a lemma about that definition.
- **So "the ledger is whole" is not proved without naming `Numbered`**, here or in the next pull request.

## What the statements do not say

- **Nothing ties them to the pooled integrity statements.** `Flock.Guarantees.IntegrityPeriod` and the two beside it are of every sampling block of a period's run. That the record presented holds every committed window is what `CommittedOnce` states; no statement reads the two together. `IntegrityPeriod`'s docstring now says so in one paragraph (**"Every committed window is presented"**): a docstring only, no statement and no record changes.
- **What one-stage v2's accept means.** `AcceptMeans` holds for any audit of one registration; it says the check called it once a window on the right inputs, not what its accept gives.
- **A session that cannot be sampled is published, not audited.** Its layout does not hold the sample's family; nothing is held of it.
- **A committed root is a v2 open of the lane inside the period's run of entries.** An open under an older schema is not one. The run's two ends are read from the device's stamps, of which nothing is taken.
- **Windows the check refuses**: a session that sampled several registrations, and a registration whose public file's header names `shared_rows`.
- **A line's signature is not held to the device's.** The device's own line and the presented one say the same and may be signed apart.
- **The history must hold the device's whole ledger under the key, from its first line**, for the two ledger statements to say anything of a device: a line the history lacks is a forgery to the finder.

## The lock

`security/Properties/lean-audit.json`, by hand:

- **`roots`** gain the two modules; the lakefile's too.
- **`layers`**: `Properties.Spot.PeriodLedger` gets its own layer, since it reads `Properties.Challenger` beside `Spot` and the period's statements. It imports nothing of Mathlib or `security`, and no proof.
- **`shape.frozen`**: one `unstated:` line for each of the five. The lock's rule fails a `Prop` in the statements tree that is neither a listed guarantee's statement nor held there with its reason.
- **No guarantee entry, and why.** A guarantee entry here is keyed by the theorem that proves it, and a property of an acceptance needs an honest witness: a ledger whose LMS signatures the kernel verifies, which the check accepts. Neither exists in this pull request. `LedgerOfDevice`'s line says it gets no entry while it takes `Numbered`.

## The executable

`sampled-proofs-period`: `Spot/PeriodMain.lean` is `Core.BlobProtocol.main` over `Spot.Period.protocol` (`id`, `check PARAMS COMMITMENT EVIDENCE [BLOB ...]`, `serve`). It is a `lean_exe` of SPOT's verifier and one of its default targets, and its entry point is `exempt` in the verifier's lock as each entry point is.

- **The difftest asks it** the two cases the piece is asked on one-stage v2 itself, through `serve`, and holds its answer to the protocol's outcome for the reference's reject, under the SHA-512 of the ledger's bytes.
- **It is asked no accept**, as the check is not: no proof in the tests is one that one-stage v2 accepts.
- **Nothing calls it.** No gate, no demo and no configuration runs it yet.
- **Its name** follows the package's other executables. It counts against the cap on SPOT's older name (bookkeeping, a warning); `spot-period` would not.

## Checks

- **The five statements elaborate**, on `propext`, `Classical.choice` and `Quot.sound` only: `r20261112-222731-6f65`, at `d3825441e7cf`. Two of them changed after it (`AuditedAt` names the header's `shared_rows`; a line's text is the text its message hashes). The next pull request's proofs elaborate against them as they stand here: `r20261113-012610-3281` and `r20261113-014041-b335`, on its branch.
- **The executable builds and answers**: `r20261113-020207-0c67`, exit 0, at `91d05f87f581`. `lake build Spot sampled-proofs-period sampled-proofs-difftest` (504 jobs); `sampled-proofs-period id` prints `verity/sampled-proofs/period/v1`; both period difftests, 24 passed in 130 s. The head after it changes docstrings and one comment.
- **On the head** `43fde9d04680`, by hand: `tests/test_lean_packages.py`, `tests/test_retired_terms.py` and `tests/lint/test_lean_bar.py`, 65 passed; `modules.py --check` on the new module.
- **Not run yet on the head**: the records run and the quick tier. No ready line until both are in.
- **A second reader** is on the five statements; none has reported.
