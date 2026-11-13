Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. The third of its pull requests: what an accept of SPOT's period check means and whose ledger it reads, as eight statements, with the check's executable. It is on the audits' half's head (#9595), which is on #9537's; both land first.

**Nothing here is proved, and no statement gets a guarantee entry.** The proofs are the next pull request (#9625). Each statement is held in the lock as a draft (`unstated`), the way the three drafts of `Properties.Challenger` are.

## The statements

**`security/Properties/Spot/Period.lean`**: six statements about the check's functions as they run, for every input their decoders produce. They take no premise. Each is about the ledger the check was presented and says nothing of whose it is.

- **`LedgerMeans`**: what the ledger's half returns (`windowsOf`). The ledger verifies under the verifier's key and reaches the monitoring period's end. Every root the lane committed in the monitoring period is returned: a window when its session can be sampled, one of the other sessions when it cannot. Nothing else is returned, and no root and no session is returned twice. Each window's session has ended, in a closed period all of whose sessions are windows.
- **`AcceptMeans`**: what the check accepts (`checkWith`, at any audit of one registration; the check itself is at one-stage v2's). It publishes this registration's circuit, the ledger's half's windows in order and, of its other sessions, what `registeredIn` returns. Each window has exactly one audit, and that audit is the check's own at the window (`AuditedAt`): the session is the one the ledger sealed, run from a commit the pins log records on main's first-parent line, with its coin commitment withheld; its log shows the sample's coins released after exactly one registration; and the audit accepted that registration at the parameters the check composes for the window, after the records of the windows before it.
- **`CommittedOnce`: every root committed under this registration is one published window.** Under an accept, every root the lane committed in the monitoring period is on a session that a period committing a circuit lists. Where that circuit is this registration's, the root is the root of exactly one published window, which has one audit entry. Each published window is such a root.
- **`OthersElsewhere`: the published others are exactly the lane's roots under other registrations.** Under an accept, every root the lane committed in the monitoring period under another circuit than this registration's is published once among the others, by that circuit and its session, and each published other is such a root.
- **`DecodedMeans`**: what the decoder gives the check (`AuditParams.of`). The key each audit is handed reads as the key the ledger is checked under, and `audit` names none of the keys a window sets or leaves unset.
- **`AuditedSession`**: the session audited at a window is the window's own. The ledger opens a root once, so the session `AuditedAt` reads on a window's root is the one the window names.

**`security/Properties/Spot/PeriodLedger.lean`**: two statements of whose ledger the check reads, against the challenger's model of the device (`Properties.Challenger`: `History`, `Break`).

- **`LedgerLinesOfDevice`** takes nothing of the device. Every line of a ledger the check reads is an entry under its place's number that says, its signature aside, what a line of the device's own ledger under that key says; or the fixed finder outputs a break (an LMS forgery or a SHA-512 collision).
- **`LedgerOfDevice`** takes one hypothesis, `Numbered`. Then the presented lines are, their signatures aside, the first lines of the device's own ledger, in order, none dropped and none added, up to a line stamped at or after the monitoring period's end; or a break.

## Whose a root is

The statements read the check that holds the refusal (#9595). The ledger ties a session to one digest: the circuit that the period listing it committed (`registrationOf`). A root is committed under this registration when that digest is the verifier's parameter `registration`.

- **There is no clause for other sessions in `CommittedOnce`, and no corollary at an empty list of them.** The check refuses a root committed under this registration without the sample's family, so every such root is a window.
- **A root under another circuit is published, and nothing is held of it**: not that its circuit is a registration, and not that any check audited it.
- **Nothing ties `registration` to the circuit that `audit` describes.** The verifier supplies both; one-stage v2 reads no period's circuit.

## `Numbered`: the one hypothesis, and why the second form needs it

`Numbered dev pk` says each entry of the device's ledger under the key is the line at its own number: a line of it that is an entry numbered `i` is its line `i`. It asks nothing of a line that is no entry the reader takes.

- **Without it the ledger is not shown whole.** A device that wrote two lines under one number satisfies `LedgerLinesOfDevice` while the presented ledger holds only one of them.
- **The device does number its lines.** `verity/protocols/spot/device/challenger.py` writes each line with `"seq": len(lines)`. The model of the device (`History`) does not hold that yet.
- **It is a hypothesis of `LedgerOfDevice` alone, not a named assumption.** No lock entry reads it. Its docstring says it goes when the device's definition carries the numbering: then it is a lemma about that definition.
- **So "the ledger is whole" is not proved without naming `Numbered`**, here or in the next pull request.

## What the statements do not say

- **Nothing ties them to the pooled integrity statements.** `Flock.Guarantees.IntegrityPeriod` and the two beside it are of every sampling block of a period's run. That the record presented holds every window committed under this registration is what `CommittedOnce` states; no statement reads the two together. `IntegrityPeriod`'s docstring says so in one paragraph (**"Every committed window is presented"**): a docstring only, no statement and no record changes.
- **What one-stage v2's accept means.** `AcceptMeans` holds for any audit of one registration; it says the check called it once a window on the right inputs, not what its accept gives.
- **Two registrations of one lane under one family.** A window whose period commits another circuit refuses the period.
- **Who the lane is.** A committed root is a v2 open inside the period's run of entries whose body carries the lane's name. Who may reserve under a name is the device's gate's, and an open under an older schema is in neither list.
- **The order of the windows, and time.** A window's phase is its place among the windows. The run's two ends are read from the device's stamps, of which nothing is taken.
- **What one-stage v2 holds through `seed_salt`, `day` and `declarations`.** The check sets none of them at a window.
- **A registration under another family** on a window's session is passed over.
- **Windows the check refuses**: a session that sampled several registrations, and a registration whose public file's header names `shared_rows`.
- **A line's signature is not held to the device's.** The device's own line and the presented one say the same and may be signed apart.
- **The history must hold the device's whole ledger under the key, from its first line**, for the two ledger statements to say anything of a device: a line the history lacks is a forgery to the finder.

## The lock

`security/Properties/lean-audit.json`, by hand:

- **`roots`** gain the two modules; the lakefile's too.
- **`layers`**: `Properties.Spot.PeriodLedger` gets its own layer, since it reads `Properties.Challenger` beside `Spot` and the period's statements. It imports nothing of Mathlib or `security`, and no proof.
- **`shape.frozen`**: one `unstated:` line for each of the eight. The lock's rule fails a `Prop` in the statements tree that is neither a listed guarantee's statement nor held there with its reason.
- **No guarantee entry, and why.** A guarantee entry here is keyed by the theorem that proves it, and a property of an acceptance needs an honest witness: a ledger whose LMS signatures the kernel verifies, which the check accepts. Neither exists in this pull request. `LedgerOfDevice`'s line says it gets no entry while it takes `Numbered`. `DecodedMeans`, `AuditedSession` and `OthersElsewhere` are lemmas a reader takes beside the others, no guarantee's statement.
- **`OthersElsewhere` stays, and the count is eight `unstated` lines.** It states soundness's own sentence of the refusal: "others" must be exactly "other registrations", and that is checked.

## The executable

`sampled-proofs-period`: `Spot/PeriodMain.lean` is `Core.BlobProtocol.main` over `Spot.Period.protocol` (`id`, `check PARAMS COMMITMENT EVIDENCE [BLOB ...]`, `serve`). It is a `lean_exe` of SPOT's verifier and one of its default targets, and its entry point is `exempt` in the verifier's lock as each entry point is.

- **The difftest asks it** the two cases the piece is asked on one-stage v2 itself, through `serve`, and holds its answer to the protocol's outcome for the reference's reject, under the SHA-512 of the ledger's bytes.
- **It is asked no accept**, as the check is not: no proof in the tests is one that one-stage v2 accepts.
- **Nothing calls it.** No gate, no demo and no configuration runs it yet.
- **Its name** follows the package's other executables. It counts against the cap on SPOT's older name (bookkeeping, a warning); `spot-period` would not.

## Checks

- **The statements build** as they stand here: `r20261113-035417-2795` on vy-nebius-cpu-11 built `Properties.Spot.Period` and `Properties.Spot.PeriodLedger`, on the proofs' branch, whose statement files and verifier are this head's.
- **The proofs elaborate against them** (#9625): `r20261113-035213-962c` and `2795`, on `propext`, `Classical.choice` and `Quot.sound` alone.
- **The check itself at #9595's head**: `r20261113-034210-2101`, both period difftests, 28 passed.
- **The executable builds and the difftests pass at this head**: `r20261113-040702-87ae` on vy-nebius-2. `lake build Spot sampled-proofs-period sampled-proofs-difftest`, 504 jobs; the executable answers `id` with `verity/sampled-proofs/period/v1`; both period difftests, 28 passed in 153 s. No ready line before #9595's.
- **No quick tier and no Lean check on this head.** The lock's eight lines are by hand; `tests/test_lean_packages.py` passes on the proofs' branch, which holds them.

Refs PRF-600
