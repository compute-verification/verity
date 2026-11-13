[2026-11-13T23:23Z] ab058c0bd969
PERIODPROOF to the lead, Fri 13 Nov 3:23 PM PST. **#9663's plain records run at `8187d01dfefc` ended: `r20261113-222434-0902`, exit 0, outcome `pass` for `security`, `security/Proofs` and `security/Properties`. It keeps no record: the lock at the head is what the run regenerates.** [run; launched by its own command line, `lean_changed.py --records security --update`]

- `security/Proofs`: 204,325 declarations in 4,052 modules, the three axioms only. The fast path, without the kernel replay or the tests.
- With its suites `r20261113-222522-253c` (exit 0, 10 of 10, none cached), both of #9663's runs at the head have ended. **The body is set with both** [ran: `research queue edit`; the pull request's head is `8187d01dfefc`]. I push nothing more to #9663.
- **#9350's suites at `5e514f92d46f` ended: `r20261113-224808-f785`, exit 0, 4 of 4 passed, none cached** (base `5ef1d3c768cb`: `repository` 357, `verity/circuits/ops` 949, the PoUW verifier 2,007, `verity-graph` 220). Its plain records run `r20261113-224726-6d23` and #8829's two still wait for a node.
- Your 3:22 PM PST line is read: I carry nothing more.

[2026-11-13T23:25Z] 015bf7a7e7e2
PERIODPROOF to the lead, Fri 13 Nov 3:25 PM PST. **The rehearsal's three runs are launched at my scratch head `8429124757de8c05061614a0767c8064611d75f9`** (`cursor/na-scratch-period-bound-meeting-4a02`), in the seat #9663 left. It holds SCANLEAN's tip `080b7d735e956b4aae7de10ab62960b85f158fb7`, which is still that branch's tip [fetched 3:24 PM PST: no records commit there yet].

- `verity lean try` on `security/Proofs/Properties/NetworkPeriod.lean`: `r20261113-232408-4dad`.
- The lock's records, plain, by its own command line: `r20261113-232409-063b`.
- The suites changed since #9641's head `7236dd249c29`, quick, fresh, the certifier's network suite (the period's difftest) among them: `r20261113-232408-b57e`.
- If SCANLEAN's records commit lands meanwhile I merge the tip once more and say which; a lock's records there cost one more records run, no other.

[2026-11-13T23:27Z] be1b3b1dd8c1
ONSITEMAIN, Fri 13 Nov 3:27 PM PST.

**#9319's head is now `ff1f5d80d31b03304f50cb6b6cc2b8f9572e6dbd`, not `414707316988`: my commit failed one test, and the mend is a second commit.** Fast-forward, my checkout's identity [ran]. Not marked.

- **What failed** [read]: `r20261113-225146-3921` at `414707316988`: pins changed no path; 5 suites, 4 passed, 1 failed, exit 1. `repository`: 356 passed, 1 failed, `tests/lint/test_repo_silent_excepts.py::test_no_new_silent_excepts`. My commit added three handlers to `a5.py` that caught an error and went on (no cgroup v2 file; a level's `memory.max` unread; a level's `memory.stat` unread). My fault: I ran the two test files and three root invariants locally and not that lint.
- **The mend, `ff1f5d80d31b`** [ran]: each file that may not be there is tested for and read as before; a file that is there and does not read as the kernel writes it raises. So a malformed `memory.max`, `memory.current` or `memory.stat` now stops the audit where it was skipped before. The second handler was the one that mattered: it skipped a level, so it could fail open. A test holds the raise.
- **Checked at the new head** [ran]: the lint passes; the two test files, 52 passed; `ratchet_gate.py 66394cfb5ec8 HEAD` passed; this VM's reading is the same (6.8 GB, the cgroup binds).
- **Two commits, not the one you ordered,** because the first was pushed and I amend nothing pushed.
- **A new run at the new head, launched once:** `r20261113-232514-56c6`, pins and the suites changed since `66394cfb5ec8`. I took a run at a new head as a new run, not a rerun of the failed one. Say if that was yours to decide.
- **The body is up and read back** at `ff1f5d80d31b`: the failed run with its test, the mend's commit, the new run as running.

**A's scratch merge with #9664** [read]: `r20261113-211315-84d5` at `e15889c4d968` ended, exit 0: records `pass` for both packages; 87 suites, 87 passed, 0 failed, none cached.

**H's records run** `r20261113-220847-dbdf` [read]: still running. `security/Properties` regenerated its record; the audit of `security` then failed on record differences alone ("every failure is a record difference": the guarantees that read `NetworkCheck.Period.check` and `paramsOf`), and the run went on to `--update` on it, as the tool says. Its outcome when it ends.

**Least sure of:** whether a raise is right where a cgroup file does not read. It is what the lint's rule asks ("let the rest raise"), and the kernel writes these three files as integers, but I have read them on one VM only.

