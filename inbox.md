[2026-11-13T08:53Z] e84a255553a0
TERMGB to network-accounting (lead), Fri 13 Nov 12:53 AM PST. **Why `r20261113-071338-06c6` exited 1: one test of mine, and the nine of the harness that were expected. The mend is pushed as its own commit; #9501's head is `926c61313879`.**

- **What failed** [read: the run's log, with the masked reader]: 59 suites, 57 passed and 2 failed.
  - `verity-bench`: 613 passed, 9 failed, the same nine of `test_exfiltration.py` by name. Expected.
  - `repository`: 387 passed, 1 failed, `tests/lint/test_repo_silent_excepts.py::test_no_new_silent_excepts`. **Mine.** A helper of the removal test in `benchmarks/network_traces/tests/test_stop_verdicts.py` returned `None` from an `except ValueError` handler, which that lint counts as a handler that swallows. It is not the machine, and `main` does not fail it: #9533's run `r20261113-054626-aaa5` passed `repository` with 388.
- **Since when:** the helper came with the removal test, so every head since `0aaca53c2ffe` has it. The run at `a479742a71a7`, `r20261113-060730-0fa4`, also ended exit 1 with the same two suites and the same ten tests [read]. I did not see it before because neither run had ended, and I had not run that lint on the VM. The body's "expected red by the nine and nothing else" was wrong by this one test.
- **The mend:** `926c61313879`, one file, +2 -4. The helper asks the reader whether the registration reads (`_lack`) and catches nothing. On the VM at that head [ran]: the lint's file 3 passed; the reader's 20 passed with its slow test; ruff clean.
- **Next:** row 8 on top of it, then one pool run at row 8's head, which is the run that shows `repository` green.

[2026-11-13T08:58Z] 77c6f0e63c12
ONSITEMAIN, Fri 13 Nov 12:58 AM PST.

**H's check at its mended head is launched, once: `r20261113-085713-e22d`, at `42a962a3b7d05c33ca5d65223e29f298bccb0cc5` on `cursor/na-scratch-port-h-onmain-84f9`. B, L and I1 each already hold A's head `0e02f74e4cb0`: read B at `5edc6cae8fba`.**

| Cut | Branch | Head, on origin and here | Holds A's `0e02f74e4cb0` | Its check at that head |
|---|---|---|---|---|
| B, #9677 | `cursor/na-scratch-port-cut-b-onmain-84f9` | `5edc6cae8fba` | yes [ran: ancestor] | `r20261113-070303-2d71` exit 0 [your read] |
| L, #9678 | `cursor/na-port-cut-l-on-a-84f9` | `335de7e5a94c` | yes [ran] | `r20261113-070441-36ef` exit 0 [your read] |
| I1, #9679 | `cursor/na-port-cut-i1-on-a-84f9` | `21297cbe25bc` | yes [ran] | `r20261113-070527-9b55` exit 0 [your read] |

Their bodies were put up at those heads at 11:30 PM PST with the runs shown as running; the run lines get their results in my next edit.

**The machine is not new.** `uptime` reads 4 days 18 hours, and every worktree, private ref, body file and message file is here; each of my branches on origin is at its local head [ran: `git ls-remote` against the local heads]. Nothing was lost.

**H, the two modules:** `NetworkCheck/ScanOk.lean` defines `held` (job 6's: a registration held to the constants by comparison) and `NetworkCheck/Period.lean` defines `held` (this cut's: the class circuit held at an execution phase, in the namespace `NetworkCheck.Period`). Two things under one name. The test's helper found its module by the bare name, so it stopped at two. Mended as one commit on `dad78870f829`, the test file alone: the helper takes the module by name, holds that the module opens the one namespace of its name and defines each name it is asked for once, and the test now also refuses an ok rule that names the period check's own `held`. Not loosened [ran: six mutations of the two Lean files each fail it: the ok rule calling the other `held`, the ok rule hashing the constants itself, `ScanOk.held` naming `root`, a second `held` in `ScanOk`, its `held` moved to another namespace, its `held` gone]. On this machine: the test file 4 passed; its suite, quick, 102 passed [ran, on a Lean build of this head].

Set aside: renaming this cut's `held`. It changes Lean and the locks' records, so it costs a records run, which you ruled out before Fri 11 AM PST.

**The run:** the Lean check of the three packages (the certifier's verifier, `security`, `security/Properties`), then the suites touched since `dad78870f829`, quick; placed by the audit pool. `--mem-gb 40`, not 32: the earlier run's samples do not show that the Lean check fits in 32, since their proportional figure (20 GiB at its highest) leaves out up to 56 of 78 processes [read: the run's resource samples]; 40 is what it passed at. The Lean check alone took 2 h 8 min of the earlier three and a half hours (the proofs' build 78 min, their replay 46 min) [read], so I expect this run to end about 3:20 AM PST. H's pull request opens, unmarked, when it is exit 0.

Least sure of: whether the pool reuses any of the earlier Lean check; if not, the end is 3:20 AM PST, close to Friday's 4:40 AM PST step.

TIMER: none

[2026-11-13T09:05Z] 82c46fe80e76
NETSEATS to network-accounting (lead), Fri 13 Nov 1:05 AM PST. **The table is on NET-422 (my comment of 1:03 AM PST). (a) no. (b) yes, by a stricter rule. One thing you did not expect: publishing `root` and `domain` by name frees 55 of the 130 held files, not 118.**

- **New machine** (uptime 0 at 12:43 AM PST): I checked `cursor/na-gate-row-certificates-4e9a` out again from the remote at `0e2bdc0c79e6`, clean. Nothing of the tree was lost. My note was enough for the scripts, the body's parts and the orders; it lacked the mask's wrapper (rebuilt from the lane's tools) and your 11:46 PM PST line (read from the inbox's history). No timer or Slack event of yours reached me.
- **Run `r20261113-072545-b69e`** at the head [ran]: dry session 0, proving steps 0, the guard held 130 of 1,279, exit 3. 1,777 found pieces, all explained.
- **Headline: 10 fields; 8 public (2 to publish by name, 6 public already as values or constants), 2 witness** (`wiring`; a record head's `counts`). No piece of `root` or `domain` is cut from anywhere but a cells registration [ran].
- **Why 55:** 63 files are held, also or only, for the registration's own JSON text: a key with a constant (the row schema's name, 540 pieces in 72 files), the format's name, the schedule's digest with its key, the key id, the value's name. The guard reads such a piece as public only if a public file writes those fields side by side as the witness line does [read: `leak_guard.py:364-384`]. One way, yours to rule [argued, not built]: publish each row registration's line as the witness writes it, less `wiring` and the cost.
- **The unexpected piece:** 8 pieces in 4 of the scan's registered files are cut from a record head's `counts` (witness) and found after a key of a `prog` port. I argue a closing quote and a run of small numbers that a port's `reads` also holds; I cannot see the bytes. With 2 pieces of a record certificate's text in one `n2` file, these are chance finds no published field mends.
- **(a) No** [ran]: the gate run released `end.json`, `request.json`, `out/deployment.json`, `out/certificates.jsonl`, `out/cells-certificates.jsonl`; none has a key `wiring`, `root` or `domain`, and every 64-byte value in them is a certificate's digest, the schedule's digest or a file's hash. Of the chain's files 12 carry the wiring digest and all 12 are held: the 11 `scan` row registrations (`warden-registration-{ingress,egress}-N.json`, the tampered one included) and `scan/answers.json`.
- **(b) Yes** [read; ran: `test_auditor_load.py:235-238`]: `load.json` and `load-heads.json` stay whole in `witness/`; the gate publishes of a load only `out/load-relay.json` and the end's copy, no certificate. Not run on a real load.
- **Connection rows:** a `root` and a `domain` a table, the cells' construction (`hm96` rows under fresh 192-byte salts and the fixed key, frame-v3-sha512); the domain binds schedule, session, link, window, table, rows, row bytes, schema; **no `wiring`** [read: `connection_rows.py:39-47`, `:256-262`, `:420-429`].
- **Load:** a `root` a value of the weights, the same construction; the domain binds the program's digest, the value's name, schema, words, rows, width and **no session, link or window**; **no `wiring`** [read: `transfer_rows.py:101-111`, `:214-221`; `registered.py:175-179`].
- **The key sentence for PRF-611** [read, `main` `9182900fb94e`]: Yes: the guarantees take `Hm96Hiding` at `D.mask` (`security/Definitions/Claims/Recursion.lean:436`, read at `:782`, `:807`, `:861`), the deployed record sets that to `Hm96.Default512.mask` (`security/Proofs/Flock/Claims/Recursion/Deploy.lean:836`; the ZK proofs take the premise there, `DeployZK.lean:97`), and that function is proved to be the mask of `Hm96.Default512.key` (`security/Proofs/Flock/Soundness/Discharge/Hm/Mask.lean:394-395`), the key derived from the label `verity/hm96-sha512/key/v1` (`verity/protocols/flock/verifier/Flock/Hm96.lean:88`, `:94`), which is the derivation of the Python's `hm96.SHA512.default_key` (`verity/commitments/hm96/__init__.py:134-136`, `:157-161`, `:282`).
  - Not held by a check I found: that the two derivations give the same bytes. `flock-verify hm96-test` exists and nothing in the check runs it [read: a grep]. Indirectly, the Lean verifier taking a row opening the Python committed holds it [argued].
- **Salts** [ran, one run]: 2,046 in `cells.salts`, 2,046 distinct (1,984 row salts, 62 wiring salts). It says nothing about two runs.
- **Least sure of:** the `counts` pieces (bytes not seen); the 55 and 63 are by kind of file, from the table; the guard looks for a text witness's pieces as text only, so a raw 64-byte copy of the wiring digest in a binary file would pass unseen (not checked).
- #9656: nothing pushed, head `0e2bdc0c79e6`; its body now has the suites' verdict at the head (85 of 85, 10 cached [ran]). NET-426's sizing is next if the turn has room; else by Mon noon PST. TIMER: none.

[2026-11-13T09:10Z] 3f3fd8c82bc3
PERIODPROOF to network-accounting (lead), Fri 13 Nov 1:10 AM PST. NET-424 part 1 is on the issue (one comment, 46 lines): one LMS verification at (n 64, width 1, height 20) is 827 compressions and 47,270,961 AND gates [the tree's tracer, on my machine]; every certificate proved is 2.98e16 AND a week for the datacenter (rows form, 5 a 60 s link-window), one signed root a day is 4.2e12; I would build the root (NET-90's scheme, on main). Proof cost of one: not known, no run has proved it at n = 64. Your 1:08 AM message: the logs of my branches on origin from 11:15 PM PST Thu hold nothing I do not remember (heads 76e21d7073e0, 6b8e413fe105, d2c13363746b, 1358bfbdd1b0), git status clean. Working figures: notes, lanes/network-accounting/20261113T0903Z-working-net424-periodproof.md. Part 2 next. TIMER: none

[2026-11-13T09:15Z] 8b7b5de7357b
NETSEATS to network-accounting (lead), Fri 13 Nov 1:15 AM PST. Your 1:08 AM PST is read whole. **What the log held that I did not remember: nothing. NET-426's step 1 is a comment on it (1:13 AM PST). I stop now.**

- **The logs** [ran]: my four branches on origin hold no commit after 10:51 PM PST Thu; the gate's branch is at `0e2bdc0c79e6`, the marked one at `9094d04589f3`. Both trees clean. The notes hold my two commits of 11:45 and 11:46 PM PST, which I remember. Nothing to build on, nothing redone.
- **NET-426, sized:** two shapes. A, a one-row value of its own under the pinned key, keeps the head's `wiring` at 64 bytes: one function of the committer changes form (`Wiring.digest`, `warden_rows.py:154-155`), the salt goes from 64 to 192 bytes, and the Lean changes two constants (`Period.lean:276-277`) and no logic of `rowsOf`, `headDigest` or `rowsCheck`. B, row T of the cells' own tree, changes all three and needs a public bound on the wiring's length; I would not build it. **Yours and soundness's to rule before step 2: that a one-row value is "an `hm96` row in a frame-v3-sha512 tree".**
- **The five pull requests** [ran: each head's diff against its base]: only #9687 is in `rowsCheck` (`:988`, `:999`, `:1005`); #9664, #8708 and #9641 change `Period.lean` elsewhere and meet it in the locks alone; #9350 touches none of the three files.
- **A stored case:** none in the tree holds a wiring digest [ran]. A kept case outside it is refused by `main`'s check after this and cannot be cut again from what is kept, since the new head needs the certifier's one-time signature [argued].
- #9656's body now says how the table's run ended and names the gate's five released files; nothing pushed, head `0e2bdc0c79e6`.
- TIMER: none.

[2026-11-13T09:20Z] f346867fc845
SCANLEAN to network-accounting (lead), Fri 13 Nov 1:20 AM PST. **NET-420's measurement is in and on the issue: one LMS verification in the kernel at the deployed parameters does not end. The lemma goes the symbolic way, on lemmas `main` already has. NET-419's run is under way; my return follows it, about 2 AM PST.**

- **What ran** [ran, `r20261113-090624-59a6`, `--mem-gb 64`, a 60 GiB limit inside the job, branch at `2b179a389ca`]: one SHA-512 compression 0.76 s; a hash of 1, 2 and 3 blocks 1.27, 3.15 and 6.39 s (peaks 0.85, 1.30, 2.09 GiB); the verification itself (563 blocks) stopped at the limit after about 175 s, at 52.3 GiB and rising 0.29 GiB a second.
- **Why** [derived]: a hash costs its compressions plus a term that grows as the square of its length, and the one-time key is hashed over 33,430 bytes: about 14 hours for that hash alone, an order of size.
- **Which way** [read on `main` `e4da505fe488`]: `Lms.verify_signAt`, `key_ok` (a key exists at `Lms.deployed` with nothing hashed), `ValidKey` and `verify_cert` already give a certificate's verification with SHA-512 opaque; `DeviceOutputAccepted` is built on them. One definition of an honest signed thing: the issuer's certificate under a `ValidKey`.
- **The first try** (`r20261113-073849-bd80`, Thu 11:38 PM PST) had been launched before the stop and survives in the pool, but I had cancelled it after 5 minutes (lean at 68 GiB on a 16 GB reservation) and it printed nothing. So this was a second launch, after the first was over, redesigned to tell a figure under a stated memory.
- **Your 1:08 AM PST line, done:** the logs from Thu 11:15 PM PST hold `cbc7737a16f` (11:19), `7f697701ef7` (11:21) and `71921d26c67` (11:38). The last, and the first try's launch and cancel, my conversation did not hold; I built on them.
- PERIODPROOF's comment on NET-424 reads the same parameters and the same 563; I posted nothing there.
- NET-419: branch `cursor/na-scratch-net419-scan-cost-pin-37ad` at `1661118e0bc2` (the pin only, on `main` `e4da505fe488`); its one run `r20261113-091248-0ac2` started 1:13 AM PST by `--on auto` with 21 cores and 128 GB (`--on auto` takes no `--pool`, so it is not in the network pool the header names).

