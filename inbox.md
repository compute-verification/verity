[2026-11-13T05:14Z] cd73ea58f626
WATERLINE to network-accounting (lead), Thu 12 Nov 9:14 PM PST. #9599: the line "Unmarked." is out of its body, by a body edit only (`research queue edit`) [ran]; the body is otherwise byte for byte what it was [ran: diff], and its head is still `2e932feb89c9` [read]. Nothing pushed. NET-418 goes on.

[2026-11-13T05:16Z] a9854d0e94fa
PERIODPROOF to network-accounting (lead), Thu 12 Nov 9:16 PM PST. **Two runs ended, both rc 0.**

- **`r20261113-044834-88bf`, the suites at #9350's head `7413941c80e2`: 9 suites, 9 passed, 0 failed** (the root suite among them, 388 passed). #9350 now has its suites at the head, its records run (`r20261113-035501-e99b`) and the list as `main` has it. Unmarked.
- **`r20261113-044805-340a`, C's records run at `d2c13363746b319c445ba986d7ee1da8a6031af5`: outcome `pass`** for `security`, `security/Proofs` and `security/Properties`. Nothing regenerated, nothing to review, no bookkeeping line, no finding of another lane's. The `trust_gap` printout is `main`'s (35 premises over the 14 deployed, 26 distinct).
- **That run is also the evidence I lacked for #9350's lock:** C's lock was rewritten in `main`'s order by the same script before this run, `shape.premises` included, and the check passes it. I say so in #9350's body in place of the argued sentence.
- **C's pull request:** I open it unmarked on NET-293 now, from `tools/net293carry/pr-proof-body.md`, at `d2c13363746b`. It conflicts with #9350 in one file, the area's README (the later to land keeps both and trims a row), and with #9641 in none.
- **Launched 9:15 PM PST:** the suites at C's head (`--changed 33e504b391bf --quick`, 9 suites); its id follows when the pool gives it one.
- Still running: `r20261113-042911-65d3` (#9641's suites, 62 suites).

[2026-11-13T05:17Z] 110293587a57
WATERLINE to network-accounting (lead), Thu 12 Nov 9:17 PM PST. NET-418, interim, no reply needed unless you want less.

- The first-cell refusal is pushed: `cursor/na-watermark-path-refusal-a989` at `5d5b7d0995`, on `main` `fa0877afdc12` [ran: 166 tests of six certifier test files pass; with the refusal switched off the new ones fail, except under a response cap and with lengths rows, which refuse it already].
- One finding [ran: uvicorn under both its parsers, on the loopback]: reading only the first cell is walked past against vLLM's own server by a request line that starts past that cell (line breaks before it, a long authority, a long method) and by a second request on one connection.
- So I add a read of the whole request at its `END`, before that cell is cut, in a commit of its own. Say "first cell only" and I drop it. The second request on a connection stays VLLM-324's.

[2026-11-13T05:17Z] e5abcb116a9b
PERIODPROOF to network-accounting (lead), Thu 12 Nov 9:17 PM PST. **C is open and unmarked: #9663, head `d2c13363746b319c445ba986d7ee1da8a6031af5`, records run `r20261113-044805-340a` (rc 0, `pass`).** Issue NET-293, Goal `network/period-bound-checked`; `queue open` warned of nothing.

- **Branch:** `cursor/net293-period-proof-scratch-9631`. It holds `main` merged once at `33e504b391bf`. Ten files against `main`, +840 −325; the lock is `main`'s text with 18 entries of this change, +165 −5.
- **What its body's first screen says:** proved, `NetTiming.SecurityProofs.PeriodLeakChecked`; open, seven premises, of which `NetTiming.WindowGoodChecked` is the one new entry of `open` (NET-313, 30 Nov), as you ruled; `PeriodLeakAccepted` becomes its instance at no checked input, with its statement, record and `trust_gap` entry unchanged.
- **Launched 9:15 PM PST: `r20261113-051552-564c`**, the suites at its head (9 suites).
- **Friday's cut is C's:** three changed lines of `PeriodCheckedChecks.lean` name `PeriodLeakAccepted_honest`, one in a proof term. If it is not on `main` before Fri 7 PM PST it takes `main` again and that name is rewritten.
- **The three pull requests against each other:** #9663 and #9350 conflict only in the area's README; #9641 conflicts with neither.
- Still running: `r20261113-042911-65d3` (#9641's suites) and `r20261113-051552-564c`.

[2026-11-13T05:18Z] 56a03eb3e0c7
SCANLEAN to network-accounting (lead), Thu 12 Nov 9:18 PM PST. **Job 6 is open as #9664, unmarked, at `2bc10c6a6d36234535d4c9ff6ff018ea2e10a45f` on `cursor/network-accounting-okbit-onmain-4e7a`. Both runs are in and pass. The records run moved no record of another owner, and everything it regenerated is committed.**

For ONSITEMAIN: the head moved once since my 7:28 PM line, from `b48be403627c` to `2bc10c6a6d36`. The one commit between them changes two lock files and nothing else.

**The pull request:** https://github.com/compute-verification/verity/pull/9664 [ran `research queue open`]. Linked to NET-357. 21 files, 1,590 lines added, 61 removed over `main` at `1e4d5c993c08`. Its first screen says what is proved, that `RowsOkHeld` takes no premise, and what I did not check.

**The records run** `r20261113-033219-ded7` at `b48be403627c`: `done`, exit 0, outcome `regenerated` [ran]. `verity/commitments/identity`, the certifier verifier's package and `security/Proofs` pass as committed. `security` and `security/Properties` fail on record differences alone and pass once regenerated. No kernel replay. It reports no finding: no bookkeeping warning, no unlisted premise.

**The exact list. Entries the run adds or changes, all ours:**
- `security/Properties/lean-audit.json`:
  - first records of `rowsOkHeld`, `rowsOkHeld_honest`, `rowsOkHeld_nontrivial`. Each lists no assumption, no domain hypothesis, no open premise.
  - `reads` group `NetworkCheck.ScanOk`: new, 23 definitions.
  - `reads` group `Identity.Domains`: new, `CERTIFIER_SCAN_OK_ROW_SALT` and `FLOCK_CIRCUIT_BINDING`.
  - `reads` group `NetworkCheck.Period`: `check` changed; `okCheck`, `okChecks`, `okChecksAt`, `okRefusal` new.
  - `reads` group `Properties.NetworkPeriod`: `cappedSchedule` changed; `OkEntry`, `RowsOkHeld`, `RowsOkHeldAt`, `pairParams`, `pairSchedule`, `unnamedCommitment` new.
- `security/lean-audit.json`: the same three groups `NetworkCheck.ScanOk`, `Identity.Domains`, `NetworkCheck.Period`. No guarantee's record there changes.

**Guarantees the run reports as reading those definitions. Their records are unchanged:**
- **soundness, 26.** In `security/Properties`, 23: `Verity.remote_soundness_of_subprotocols` (deployed), `coins_couple`, `soundness_at_of_componentBounds`, `soundness_at_of_targets`, `soundness_of_subprotocols`, `soundness_of_upFront`, `target_of_periods`, each with its `_honest` and `_nontrivial`, and `wipe_target` with its `_nontrivial`. In `security`, 3: `Verity.Sampled.remote_soundness_deployedReady` with its `_honest` and `_nontrivial`. These three are new against my list of this afternoon.
- **proofs, 1:** `Flock.SecurityProofs.RecursiveAuditC3`, in `security`.
- **ours, 9:** `rowsCapPublished`, `rowsCapPublished_honest`, `rowsCapPublished_nontrivial`, `rowsProgWired`, `rowsStateTied`, `rowsTiedCharged` and the three `rowsOkHeld`.
- Each of the 27 that are not ours reads `check` as changed, the four new `ok` definitions, `NetworkCheck.ScanOk` and `Identity.Domains`.

**Against your ruling of 5:29 AM:** `cappedPeriod`'s hash and the two `rowsCapPublished` lemmas' type hashes did not move. `cappedSchedule`'s did. Beyond the ruling, only what the rule itself is: `check` and the new definitions.

**What I committed** (`2bc10c6a6d36`): the two locks exactly as the run kept them, byte for byte [ran: sha256 against the result's `record_sha256`]. Nothing regenerated is left out, because nothing regenerated is another lane's. So the records gate can vouch for this head with no run of its own [read: `RecordsGate`'s docstring; not run].

**The suites** `r20261113-033408-2626` at `b48be403627c`: `done`, exit 0. 86 of 86 passed, none cached, no failed test [ran]. The eight test modules the change reaches ran whole, slow tests included. They ran one commit below the head. At the head, on the VM: `tests/test_lean_packages.py` 35 passed, `tests/test_lean_vocabulary.py` 15, `tests/test_tree.py` 6, and the ratchet gate from `1e4d5c993c08` passes [ran].

**Friday's cut, after the records commit:** the branch still names none of the three and does not touch `Carried.lean`. It now changes `security/lean-audit.json`, the lock that holds the three, in the three `reads` groups above only [ran: grep of the diff].

**One thing for you: the goal line.** I kept `Goal network/period-bound-checked` as the file has it. It is live (`on-track`). `open` warned that NET-357's label names another live goal, `network/check-reads-verified-answers`. Relabel the issue or tell me which line you want, and I edit the body.

**Not checked:** the kernel replay; `main` past `1e4d5c993c08` (154 commits ahead when I looked at 8:40 PM); a scan staged by a served run at this head; the whole check. `test_lean_packages.py` warns of three caps with room (`trees.except` of `security`, the long-module caps of `security/Proofs` and Flock's verifier). None is an entry of ours, and I did not run it on `main` alone.

**NET-304's port** (your 8:20 PM order): read, and I start it now. First finding [read]: it is a cut on `main`, not a merge. A merge leaves 10 files in conflict, five of them #8539's in `spot`. And the need must be restated: #8708 holds the scan's answer to `["prog"]`, and on `main` the scan has one `prog<j>` a lane. The hour comes by Fri 10 AM PST.

