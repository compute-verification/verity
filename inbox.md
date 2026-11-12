[2026-11-12T14:58Z] 34f737817d28
TERMGB to the network-accounting coordinator, Thu 12 Nov, 7:00 AM PT. **NET-369: pull request #9329 is open and unmarked at `c77b01b8b79af859bdfd33ea7e64b150e9717977`, on `cursor/na-forged-no-proof-b4e6`. The stand-in ran at that head under both modes, exit 0 each, and prints the new lines.**

**The runs, at `c77b01b8b79a`, exits as their statuses say** [ran]

- `r20261112-144142-7045`, the stand-in under `--zk`: exit 0. The forged rank passes on `no_proof_released`; `lean_refused` is `null`, with the reason beside it.
- `r20261112-144225-5ed3`, the stand-in under `PP_PLAIN=1`: exit 0. The forged rank passes on `lean_refused`; `no_proof_released` is `null`, "the sessions ran behind no ZK server".
- `r20261112-144251-1977`, the real two-member forged test with the Lean verifier, the ZK server and the prover built in the run: exit 0, 1 passed.
- `r20261112-144330-b96d`, `suites.py --changed 2d5f7a169d85 --quick`: exit 0, 6 suites passed, 0 failed.
- The same four ran at `ce6d38906ec1`, one docstring sentence earlier, each exit 0.

**What `no_proof_released` reads**

Two files of the forged member's own session, both of which the audit already writes. Both must hold.

1. The verifier's record of the session (`session-m<k>/sessions/*/session.json`) lists no proof.
2. The prover's record of it (the `LIVE` line of `session-m<k>/prove.log`) gives `prover_stopped` opening with `FIREWALL-REFUSED`, the ZK server's own text for a message it holds back.

It has a value only where the summary says the sessions ran behind the ZK server. `prove_bundle.py` 273-275 already applies the same rule to its own tampered sessions.

**The test fails on `main`** [ran, on my VM]: with only the test files' changes on `main`, all 15 new cases fail, by signature. So I also ran `main`'s own rule on each case's records: it differs from what the tests expect in 10 of 14 cases, this head's in none. Six mutations of the rule are each caught.

**What I am least sure of**

- **The check reads that the ZK server refused, not which of its checks refused.** Any refusal by it of the forged member's session, with no proof recorded, passes. The honest arm of the same run is the control.
- **It reads `a0.run`'s log format**, since a5 keeps `prover_stopped` nowhere else. If that format changes a forged `--zk` run fails. It fails closed.
- **`NO_SESSION_FILES` sits in `records_audit.py`**, not in `a0.lean_refused`, which still counts a `session files:` refusal. That is the line you tell proofs.
- **`records_audit.py audit --forged` is told no forged member**, so any sampled member's held proof passes it.
- In this suites run `benchmarks/one_stage` shows 2 skipped. I take them to be the two-member module's two tests, which skip without the built binaries [argued; the node's log not read]. The separate run built them.

**For you**

- `queue open` warned that NET-369 has no goal label. The body's `Goal network/slice-clean` is its parent's. I changed no label.
- The change also rewrites one sentence of `benchmarks/network_e2e/chain.py`'s docstring, which stated the old rule. That file reads only the arm's `ok`. It is the sixth file.
- The ratchet gate over #9305's head exits 1 for 3 retired-term sites, each the code's own text. It is off the merge gate; I touched no allowlist.
- It merges clean with `main` at `7d296de0f3e9` and with #9237 at `9d4aec0111fb` [`git merge-tree`]. Against lean's tip `f3837f57d113` it adds no conflict to #9305's own, by count.
- #9329 does not change `served_verdicts.sh`, so the follow-on's refusal words are left for you to list, as you said at 6:55 AM PST.
- #9305: head unmoved at `2d5f7a169d85`. I wait for your word that lean's landing is on `main`.

Not run: the whole check, `circuit-check`, any Lean check, a11.

TIMER: none

[2026-11-12T15:01Z] 66148102f411
WATERLINE to the network-accounting coordinator, Thu 12 Nov 7:00 AM PST. **Pull request #9314's new head is `7bb211d1ca06`. All three are done in one push, and the suites run at that head passed: `r20261112-144601-fcba`, status `done`, exit 0. The merge of `main` was clean.** It is unmarked; I did not mark it. Marks: [ran] by me, [read] by me, [argued].

**The run**
- `r20261112-144601-fcba`, `--on auto`, launched once, from `7bb211d1ca06`: `status.json` says `done`, `rc` 0. 6 suites, 6 passed (0 cached), 0 failed, no read outside a suite's inputs. `verity-network-traces` 640, `benchmarks/one_stage` 571, `repository` 349, `verity-graph` 215, `verity-network-e2e` 70, `verity-network-sim` 21. [ran; read from `status.json`, `job.json`, `stdout.log`]
- No second kept run.

**1. `-s 130`**
- The sentence is fixed in the README, the script's comment, the fixture's docstring and two test docstrings: `-s 130` keeps at most each packet's first 130 bytes, which is every header and, where the headers are shorter, the first bytes of its content. The number is unchanged. [ran: the tests]
- The body has the line, with a count I took from the kept capture: 993 of its 2,057 packets keep some content, 31,343 bytes in all, and a packet of 130 bytes or fewer is kept whole. [ran]
- My first report's note said "each packet cut to its headers" too. Corrected there.

**2. The release order**
- I did both. The reader now takes each connection's place from its `END` cell's place among the ingress records' `END` cells, and its release bucket from that cell. [ran: the tests]
- The two orders are the same on any record the device publishes, three ways:
  - The device keeps its connections in the order it admitted them, which is their ordinals', and cuts each bucket's cells in that order. [read: `warden.py`, `_tick` and `_admit`]
  - Its audit flags a bucket in any other order. On the test's record `audit` says not accepted, "not r cells a connection in ordinal order". [ran]
  - Its commitment to a record's rows refuses such a record. With the connections reordered and the usual publisher the run stopped with "cells aren't laid out as the schedule lays them". [ran]
- The test: twelve releases, six in each of two buckets, each bucket's `END`s by falling ordinal. It sorts the device's live connections after admission and keeps the records without the rows commitment; the records are certified as usual. The reader's place of each connection equals the device's log, and its count equals the bytes that connection's server got. With the old sort, connections 2, 4, 10 and 12 are one byte off and only this test fails. [ran, then reverted]

**3. `main`**
- Merged, not rebased: `main` at `fcc1b239d0b9`, which holds `70395628ec67`. No conflict in any file. Nothing was unclean. [ran]
- `main` has since moved to `06728b3b7c`. Nothing under `benchmarks/network_traces` changed, and `git merge-tree` against it is clean. I did not merge a second time. [ran]

**The watermark's answer: I took the bound**
- Where the watermark's connection pairs with no flow, the direction from the server now prints a bound. Its least difference drops by the bytes the capture counts from the server on the largest unpaired flow, and a line says so. One test, which fails without the change. [ran, then reverted]
- It rests on pairing by counts: the watermark's flow is taken to be among the unpaired flows. Nothing checks which flow it is. [argued; the body says it]

**Also changed, so you are not surprised**
- The result has two new keys: each connection's `release`, and `watermark_answer_in_sum`.
- The body's table is the head's reader on the kept run's files, fetched from the store: the same sums and pairing. The two negative controls give the same figures. [ran]
- The meetings with lean's tip and the clock branch were not repeated at this head. The body says so.

**Slips**
1. My first try of the release-order test stalled for ten minutes on my VM before its timeout ended it. The device's task had stopped on the commitment's refusal and the test waited on it. That is how I found the refusal.
2. `research status` printed the first characters of my own suites run's launcher line once. No other process's.

Commits pushed, body replaced, notes synced. Nothing is left only on the VM.

TIMER: none

[2026-11-12T15:09Z] 62c30a1c2431
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 7:09 AM PST. **The try of `scanScopeAcc_nontrivial_of`'s move has waited 20 minutes for the build pool's Lean slot. I have not relaunched it and will not. One thing I told you was wrong: the proof is not a bare `by decide`.**

- **The run:** r20261112-144542-aee3 at `e27a7b84e739`, launched once at 6:45 AM PST. It waits for the Lean slot of the build pool since 6:48 AM PST, one run ahead of it in line, the pool's one slot held [run: `research status`, 7:08 AM PST]. I keep reading it; tell me if you would rather I stop it.
- **What I got wrong** [run r20261112-142709-8c99 at `55c29a928932`, rc 1]: the statement reads `fun _ => oneSlot.window` as you ruled, but the bare `by decide` I named fails: `Expected type must not contain free variables: 0 < (fun x => oneSlot.window) st`. One of the file's 11 theorems failed; the 4 suites that commit reaches passed.
- **The mend** (`e27a7b84e739`, a commit on top, nothing amended): the proof decides the positive horizon at its own type, `show 0 < oneSlot.window by decide`. The statement is unchanged from your ruling. A toy file of the same shape fails the same way with the bare form and builds with this one [run locally]; the real file cannot be compiled on the VM, so the node's try is its only check.
- **Pushed** is still `c3b368cc4026`, the head whose tries and suites are all green. `55c29a928932` and `e27a7b84e739` are local until the try is read; `55c29a928932` alone does not build that theorem.

The line with the branch, the head and every run follows when this try has a verdict.

