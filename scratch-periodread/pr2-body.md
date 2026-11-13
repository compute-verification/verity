Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. The second of its pull requests: the audits' half of a monitoring period's check, on the ledger's half's head (#9537), which lands first. It is the compiled check `verity/sampled-proofs/period/v1`: every root a lane committed under this registration's circuit in a monitoring period is held to one accepting one-stage v2 audit, and a root it committed under this registration without the sample's family refuses the period. It states no guarantee and adds no lock entry, and nothing calls it yet but its difftest.

**Only a window of one registration passes.** A session whose log shows the sample's coins released after several registrations (several ranks', or one registered again) is refused under its own code, `members`.

## What it adds

`Spot/PeriodAudit.lean`, in SPOT's Lean verifier, compiled, with no Python verifier beside it.

- **The commitment** is the auditor's ledger. **The evidence** is the host's pins log and, for each window in the ledger's order, its challenger session's files and one case: the registration's record, its public file's digest and one-stage v2's evidence.
- **The check** (`checkBlobs`), in order, the first refusal the answer:

| step | code |
|---|---|
| the ledger's half (`windowsOf`) | `coin`, `early`, `missing`, `outside`, `period` |
| whose each session is (`registeredIn`, below) | `period`, `outside` |
| the evidence holds as many windows' sessions as the ledger lists windows | `evidence` |
| each window's session files are the ones the ledger sealed, run from a commit the pins log records on main's first-parent line, with its coin commitment withheld | `coin` |
| its signed log shows the sample's coins released after exactly one registration | `missing` (none), `members` (several) |
| the evidence holds one case, of that registration | `evidence` |
| the parameters at the window are one-stage v2's, and the case's public file shares no rows | `names` |
| one-stage v2 accepts the case at those parameters | `audit` |

- **The period's codes** (`Spot.Period.Code`, in the ledger's half's file) gain four: `members`, `evidence`, `names` and `audit`.
- **A reject of one-stage v2 refuses the period. An undecided leaves the period undecided**, unless a later window refuses it.
- **On acceptance it publishes** the lane, the monitoring period, this registration's circuit, each window with the registration audited and what one-stage v2 published for it, and the lane's sessions that are another registration's, each as `{registration, session}` and nothing else.

## Whose a root is

The check takes one more parameter from the verifier, `registration`: this registration's circuit, the SHA-512 its lane's periods commit as their `circuit`. The ledger ties a session to a digest only through the period that lists it, so that is what the check reads (`registrationOf`).

| a session of the lane in the monitoring period | the check |
|---|---|
| holds the sample's family, in a period that commits `registration` | a window: audited |
| holds the sample's family, in a period that commits another circuit | refuses the period (`period`) |
| holds no coin of the family, in a period that commits `registration` | refuses the period by name (`period`): its root is committed under this registration and never sampled |
| holds no coin of the family, in no period, or in a period that commits no circuit | refuses the period (`outside`): its root is committed under no registration |
| holds no coin of the family, in a period that commits another circuit | listed among the others, under that circuit |

- **It accepts less than it did, and never more.** The one case that was accepted and is now refused is a session without the family beside the windows.
- **Two registrations of one lane under one family label are never accepted in one monitoring period.**
- **Not closed here**: a lane may open a period under any digest and commit roots there without the family. They are published under that digest, so whoever composes audits must require each registration's own. And nothing in the check ties `registration` to the circuit that `audit`'s parameters describe: the verifier supplies both.

## The parameters at a window

A case supplies nothing of one-stage v2's parameters that the check does not hold to the ledger or to the registration. The verifier gives one `audit` for the whole monitoring period; the check composes each window's parameters from it (`paramsAt`).

| key of one-stage v2's parameters | whose | held to |
|---|---|---|
| `window_root` | the check's | the ledger: the root the window's session is opened on. `audit` naming one is refused. |
| `challenger` | the check's | the period's own key and family. `audit` naming one is refused. |
| `log` | the check's | `audit`'s own log, then the records of the registrations the check audited at the windows before this one. Never the case's. |
| `proofs.public` | the case's | the registration: one-stage v2 holds the public file to the registration's record. `audit` naming one is refused. |
| `seed_salt`, `day`, `declarations` | nobody's here | not derived: `audit` naming one is refused. |
| every other key | the verifier's `audit` | the same at every window. |

- **A sample's coins are read from the period's own files.** The ledger, main's history, the pins log and the session's files that one-stage v2 reads coins from are the commitment's, the verifier's and the period's evidence. A case names only its coins' segment; a case naming more is refused (`evidence`).
- **Parameters one-stage v2 would refuse at every window are refused before the check**, as the request's fault.
- **A public file whose header names `shared_rows` is refused** (`names`). The check does not hold a file's references into rows its members share to the registration.

## One change outside the period's files, and no record moves

The period checks a ledger once and reads every window from the result, which `Challenger.auditedOpen` does not offer: it checks the ledger at every call.

- **`auditedOpen` is word for word what it was.** Only its docstring gained three lines.
- **`auditedIn` is a second definition of `auditedOpen`'s 24 lines after its ledger's check**, for a ledger already checked.
- **`auditedOpen_eq_auditedIn` ties the two by `rfl`**: `auditedOpen` is the ledger's check, then `auditedIn`. An edit to one alone stops the build.
- **Nothing under `security/` changes**, so no guarantee's record moves. An earlier head of this pull request restated `auditedOpen` through `auditedIn`, which moved the records of 53 guarantees that read it (4 in `security`, 49 in `security/Properties`); this head does not.

## What it does not do

- **No accept of a real proof is tested.** No proof in the tests is one one-stage v2 accepts. The check's steps are written over any audit of one registration (`checkWith`), and the difftest runs the accept path through a stand-in audit. Against one-stage v2 itself the test sees an undecided answer and a refused coin, never an accept.
- **No statement, no lock entry.** The statements come in their own pull request.
- **No executable.** It comes with the statements. The check is served through the difftest's table only.
- **Several registrations on one session.** Refused, as above.
- **A monitoring period whose windows differ in a parameter `audit` holds** is not covered: one-stage v2 refuses the windows the parameters do not fit.

## What the difftest holds

`tests/test_period_audit_difftest.py`, against a Python reference that lives in the test alone. The ledgers are signed under a key at the deployed LMS parameters, written one entry at a time as the gate writes them.

- **What one-stage v2 is asked** is what the reference composes: the parameters and the evidence at a window, on 40 sampled requests.
- **The check with a stand-in audit agrees with the reference** on 30 sampled requests, and each of 34 named cases gets its named verdict and code over two seeds: an honest period, a session sampled twice, an audit undecided or rejected, a window never sampled, two registrations on one session, a case missing, doubled, of another registration or malformed, a case naming its own ledger, a public file sharing rows or absent, a session unpinned, off main, unsealed, not withheld, with an edited log or swapped for another's, a ledger cut short, a window in no period, a period left open, a window in a period of another circuit, a session without the family in no period, in a period with no circuit and in a period of this registration, and malformed parameters and evidence.
- **Three windows pass, and each audit is handed the records of the audits before it**; a session in a period of another circuit is published among the others, under that circuit.
- **A root committed and never sampled refuses the period**, with the reason by name, in each of the three places such a session can be.
- **The check on one-stage v2 itself** answers a real sample's case as the reference does, undecided at the proofs, and refuses a segment released after another registration.

The reference and the Lean share an author.

## Checks

- **The Lean builds and the difftests pass at this head**: `r20261113-034210-2101` on vy-nebius-2. `lake build Spot sampled-proofs-difftest`, 501 jobs, with `Spot.ChallengerSession` and its `rfl` tie; both period difftests, 28 passed in 120 s. The three slowest (53 s, 38 s, 19 s) are marked `slow`; the next takes 3 s.
- **The refusal first ran** at the head before the `auditedIn` copy: `r20261113-031857-3061`, 28 passed.
- **The quick tier at this head is in line** (`r20261113-040405-1199`). No ready line until it is in.
- **No records run on this head.** That no record moves is read from the diff, in which nothing under `security/` differs from #9537's head, not from a run.
