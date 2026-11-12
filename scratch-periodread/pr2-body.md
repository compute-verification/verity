Lane proofs

Goal proofs/sampled-audits-sound

Issue: PRF-600. The second of its pull requests: the audits' half of a monitoring period's check, on the ledger's half's head (`cursor/proofs-prf600-period-ledger-3b1f`), which lands first. It is the compiled check `verity/sampled-proofs/period/v1`: every window the ledger's half lists is held to one accepting one-stage v2 audit. It states no guarantee and adds no lock entry, and nothing calls it yet but its difftest.

**Only a window of one registration passes.** A session whose log shows the sample's coins released after several registrations (several ranks', or one registered again) is refused under its own code, `members`.

## What it adds

`Spot/PeriodAudit.lean`, in SPOT's Lean verifier, compiled, with no Python verifier beside it.

- **The commitment** is the auditor's ledger. **The evidence** is the host's pins log and, for each window in the ledger's order, its challenger session's files and one case: the registration's record, its public file's digest and one-stage v2's evidence.
- **The check** (`checkBlobs`), in order, the first refusal the answer:

| step | code |
|---|---|
| the ledger's half (`windowsOf`) | `coin`, `early`, `missing`, `outside`, `period` |
| the evidence holds as many windows' sessions as the ledger lists windows | `evidence` |
| each window's session files are the ones the ledger sealed, run from a commit the pins log records on main's first-parent line, with its coin commitment withheld | `coin` |
| its signed log shows the sample's coins released after exactly one registration | `missing` (none), `members` (several) |
| the evidence holds one case, of that registration | `evidence` |
| the parameters at the window are one-stage v2's, and the case's public file shares no rows | `names` |
| one-stage v2 accepts the case at those parameters | `audit` |

- **A reject of one-stage v2 refuses the period. An undecided leaves the period undecided**, unless a later window refuses it.
- **On acceptance it publishes** the lane, the monitoring period, each window with the registration audited and what one-stage v2 published for it, and the lane's other sessions, of which nothing is held.

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
- **A public file whose header names `shared_rows` is refused** (`names`). The check does not hold a file's references into rows its members share to the registration. This is my default, and I asked the coordinator for his word on it.

## One change outside the new file

`Challenger.auditedOpen` is split for cost: the ledger's check, then the window read from it (`auditedIn`), so the period checks a ledger once and reads every window from the result. Its body changes and no statement does. Two records' `reads` regenerate for `Spot.ChallengerSession`, in `security/lean-audit.json` and `security/Properties/lean-audit.json`. One proof that unfolds it follows (`security/Proofs/Properties/Pouw/LiveSalt/Ledger.lean`).

## What it does not do

- **No accept of a real proof is tested.** No proof in the tests is one one-stage v2 accepts. The check's steps are written over any audit of one registration (`checkWith`), and the difftest runs the accept path through a stand-in audit. Against one-stage v2 itself the test sees an undecided answer and a refused coin, never an accept.
- **No statement, no lock entry.** The statements come in their own pull request.
- **No executable.** It comes with the statements. The check is served through the difftest's table only.
- **Several registrations on one session.** Refused, as above.
- **A monitoring period whose windows differ in a parameter `audit` holds** is not covered: one-stage v2 refuses the windows the parameters do not fit.

## What the difftest holds

`tests/test_period_audit_difftest.py`, against a Python reference that lives in the test alone. The ledgers are signed under a key at the deployed LMS parameters, written one entry at a time as the gate writes them.

- **What one-stage v2 is asked** is what the reference composes: the parameters and the evidence at a window, on 40 sampled requests.
- **The check with a stand-in audit agrees with the reference** on 30 sampled requests, and each of 29 named cases gets its named verdict and code over two seeds: an honest period, a session sampled twice, an audit undecided or rejected, a window never sampled, two registrations on one session, a case missing, doubled, of another registration or malformed, a case naming its own ledger, a public file sharing rows or absent, a session unpinned, off main, unsealed, not withheld, with an edited log or swapped for another's, a ledger cut short, a window in no period, a period left open, and malformed parameters and evidence.
- **Three windows pass, and each audit is handed the records of the audits before it**; a session of another layout is published among the others.
- **The check on one-stage v2 itself** answers a real sample's case as the reference does, undecided at the proofs, and refuses a segment released after another registration.

The reference and the Lean share an author.

## Checks

- **Build**: `r20261112-213330-dbcd`, exit 0, at `c41cdcd6c8`: `lake build Spot sampled-proofs-difftest` (501 jobs).
- **The proofs that read the split**: `r20261112-204826-f29e`, exit 0, at `4fa7425a19`.
- **Difftests on the head** `5b92156ab8`: `r20261112-215441-ef46`, exit 0: both period difftests, 24 passed in 118 s.
- **Quick tier on the head**: QUICK
