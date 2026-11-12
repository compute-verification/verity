[2026-11-12T15:40Z] 2443cf820575
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 7:40 AM PST. **Job 6's line, due Fri noon PST, sent early: the branch is built, tried and pushed at `e27a7b84e739`, and waits for lean's merge. Nothing is open with you.**

**Branch** `cursor/network-accounting-okbit-constant-4e7a`, head `e27a7b84e739`, pushed. Eighteen commits on lean's landing tip `f3837f57d113`, eight of them job 6's. Not opened, not marked; no `--update`; `main` not merged in; the salts unchanged.

**Runs, each launched once**

| run | at | what | verdict |
|---|---|---|---|
| r20261112-124635-9d25 | `76a4da03ddd6` | `lean_changed.py --try` on `ScanOk.lean`, `Period.lean`, `security/Properties/NetworkPeriod.lean`, `security/Proofs/Properties/NetworkPeriod.lean`; `--siblings` on three | rc 0. Each `lean exit 0`; 63 theorems, no axiom beyond `propext`, `Classical.choice`, `Quot.sound`; no name declared twice among built modules |
| r20261112-124729-7857 | `76a4da03ddd6` | the suites the change from the landing tip reaches, fresh; then the certifier verifier's two suites whole | rc 0. 84 of 84 passed; network 97 passed, verifier 1361 passed and 1 opt-in skip |
| r20261112-131035-1146 | `ac8cd26077b2` | the suites the group-of-two commit reaches, fresh; then the two suites whole | rc 0. 57 of 57 passed (`verity-network-traces` 568); network 97 passed, verifier 1361 passed and 1 opt-in skip |
| r20261112-134034-0bbe | `c3b368cc4026` | the witnesses at T = 2: the try on `security/Properties/NetworkPeriod.lean`, `security/Proofs/Properties/NetworkPeriod.lean`, `security/Proofs/Properties/NetworkScan.lean`; then the suites that commit reaches | rc 0. Each `lean exit 0`; 63 and 11 theorems, the three axioms; 4 of 4 suites passed |
| r20261112-142709-8c99 | `55c29a928932` | `scanScopeAcc_nontrivial_of`'s horizon at `oneSlot.window`, with a bare `by decide`: the try on `security/Proofs/Properties/NetworkScan.lean`; then the suites | **rc 1.** `lean exit 1`: the bare `by decide` meets the statement's free variable, 1 of 11 theorems failed. 4 of 4 suites passed |
| r20261112-144542-aee3 | `e27a7b84e739` | the same, the proof mended (`show 0 < oneSlot.window by decide`) | rc 0. `lean exit 0`; 11 theorems, the three axioms; 4 of 4 suites passed |

- **`55c29a928932` does not build that one theorem alone;** the head, the commit on top of it, does. Nothing was amended. The statement is as you ruled at 6:17 AM PST; only the proof's term differs from what I named (my 7:09 AM PST line).
- The last run waited 30 minutes for the build pool's Lean slot (6:48 to 7:18 AM PST), then ran. It was not relaunched.
- Two launches at 4:44 AM PST were refused before any job started (they raced on shipping one commit); relaunched one after the other.
- The Lean of `ac8cd26077b2` is `76a4da03ddd6`'s [run: `git diff` of the two over `*.lean` is empty]. The Lean files `ScanOk.lean`, `Period.lean` and the two `NetworkPeriod.lean` files were last tried at `76a4da03ddd6` and `c3b368cc4026`; the two commits since touch `security/Proofs/Properties/NetworkScan.lean` alone.
- Locally at the head [run]: the retired-terms ratchet gate against the landing tip passes; `tests/test_lean_packages.py` and `tests/test_lean_vocabulary.py`, 47 passed.

**The tie's Lean** [read], as corrected at 5:08 AM PST

- Lean holds a session's public output to its public file's header root: `Flock.HmOut.checkPublic`, `verity/protocols/flock/verifier/Flock/HmRow.lean` 1956-1981.
- Lean holds the roots it is given to the registration's: `Spot.Record.checks`, `verity/protocols/spot/verifier/lean/Spot/Record.lean` 109-127.
- Python stands between: `a7.statement_checks` (`benchmarks/one_stage/a7.py` 643-647) and `scan_zk.py` 1568-1570 and 1578.
- No Lean on the scan's audit path holds the tie end to end. The Lean that states it whole, `Spot/OneStageZk.lean`'s per-member form (`headerHeld` 1002, `Member.statement` 1143), accepts nothing today (`checkMembersAt_refuses`, 1262), and the scan's audit does not call it.

**What I did not check**

- The lock's own check and `--update`: both wait for lean's merge. The hand entries hold no records, and whether the lock moves only at `cappedSchedule`'s and `cappedPeriod`'s hashes and the two `rowsCapPublished` lemmas' type hashes is unconfirmed until then.
- An unconditional kernel lemma that `RowsOkHeld`'s conclusion holds of a real registration: `decide +kernel` does not reduce `ScanOk.consts 2`.
- The sibling check beyond built modules; `circuit-check`; the whole check; a scan run at this head (the staging's code is unchanged since r20261112-103307-25cb).
- Kept runs' schedules, one by one.
- One correction of my own record: my line on the tie and T = 1 was headed 4:57 AM PST and was sent at 4:54 AM PST.

**What I hold for the pull request:** the body is drafted in the notes (`20261112T1315Z-checkpoint-net357-okbit-pr-body-draft`), with the steps after the merge and the line the body owes PRF-595's header. Tell me when lean's merge is on `main`.

