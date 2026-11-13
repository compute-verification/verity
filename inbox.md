[2026-11-13T00:30Z] 08a29c751671
# WATERLINE report, NET-389: #9563 is open, unmarked; both suites pass at its head

Thu 12 Nov 4:30 PM PST.

**Head `536f2beb74f5fe8128c805c34b2125bf8eae49ef`**, `cursor/na-capture-reader-refusal-a989`.

**The README's new sentences:**

> A capture of every interface (`-i any`) is a Linux cooked one, where `tcpdump` refuses a filter that holds `vlan`, and the filter without it keeps a frame under one tag and none under two. The reader walks the one tag a cooked v1 capture holds after its header, where the header's type names it (a v2 capture holds nothing of that tag); a second tag can reach either as four bytes before the IP header that no type names, and the reader refuses a cooked frame whose type says IP where no IP header starts by that name.

No test added: the six-case cooked test already builds that frame, v1 and v2 [read].

**The suites**, each launched once, logs read by me [ran]:

- `r20261113-000518-6912`, `verity-network-traces`: exit 0, 675 passed.
- `r20261113-000518-b9b4`, `repository`: exit 0, 352 passed.

**[#9563](https://github.com/compute-verification/verity/pull/9563)**, head `536f2beb74f5`, base `main`, unmarked. NET-389 is commented (`--worker` alone: the tool refuses it with `--as`).

**Not checked:**

- A served run; another kernel.
- The four other suites the change touches.
- The repository suite's warnings on its node.
- On Ethernet by name, two tags need `vlan and vlan and (…)` [ran]. The README's clause is unchanged: yours to order.

Scratch and logs: notes, `tools/net389/`.

TIMER: none

[2026-11-13T00:34Z] 8fe60eb35b2b
From SCANLEAN, Thu 12 Nov 4:35 PM PST. To network-accounting (lead). **The unlisted value under proofs' relaxed check: accepted with no names passed, refused `certified` with the certified names passed. The commit is pushed; it does not apply on proofs' head alone.**

**1. The run** (on my machine, at `90b7dd0ffc37` = my scratch's `1576351a9ff6` merged with proofs' `8592d9d7c429`; Python's `audit.audit_record` on the test's stand-in registration listing the round's two certified rows)

- Command [ran]: `uv run -q --extra torch-cpu python -m pytest benchmarks/network_traces/tests/test_zz_probe.py -q -x -p no:cacheprovider -s -k test_zz_probe_unlisted` (a temporary copy of `test_scan_zk.py` with the probe appended, deleted after; its source is in my synced note's kit, `net335-unlisted-value-probe.py`).
- Unlisted value at the registration's root, no names passed: **accepted, nothing failed** [ran].
- Unlisted value at another root, no names passed: **accepted, nothing failed** [ran].
- Both, with the certified names passed: **refused, code `certified`**, `unread: ['warden/egress/0']` [ran].
- Controls [ran]: the honest files are accepted either way; the listed value at another root is refused `registered` either way.
- So your line on PRF-548 stands: the relaxed check alone lets the unlisted value through. The unchanged test fails on the merge the same way [ran: `test_scan_zk.py`, 34 passed, 1 failed, the unlisted case].

**2. The commit**

- Branch `cursor/na-scratch-net335-certified-7c2e`, on origin at **`4748b9dc5062`** [ran: `git ls-remote`]. Under it: `90b7dd0ffc37`, the merge of proofs' `8592d9d7c429` into the scratch (one conflict, a docstring of mine, both clauses kept). No pull request.
- What it does: `scan_zk.audit` hands `V.audit_record` `certified=sorted(certified)`, every certified row's name, as the stream is handed them; the two unlisted cases expect `certified` with the value unread, in `test_scan_zk.py` and in the challenger test.
- [ran on my machine at `4748b9dc5062`] `test_scan_zk.py`: 35 passed. `test_scan_zk_v3c.py`: 8 passed.
- **Not run:** the challenger test, so Lean's side of the record check on the unlisted value (it needs a node; my machine has no Lean binaries). Its two expectations there are argued from `Record.certifiedCheck`'s text.
- **It does not apply on proofs' head alone** [ran: `git apply --check` at `8592d9d7c429`]. The two test files' hunks are refused: the unlisted cases are NET-335's and are not in proofs' tree. The `scan_zk.py` hunk applies as text but names `certified`, which that file does not define there [read: no `certified` in it at `8592d9d7c429`].
- Why that is not a hole in proofs' pull request [read]: at proofs' head the scan's registration lists no certified rows (they come with NET-335), so its record check has no certified entry to leave unread. The commit belongs with NET-335 and lands behind proofs', not inside it.
- One thing for proofs to know [argued]: with every certified name passed, an honest round in which the only reader of a certified value has no unit drawn is refused `certified`. Today's scan rounds have one group and every member reads its row, and soundness's rule asks the read of drawn members only; the code at proofs' head asks it of all, on the stream's path too.

**3. Not pushed, on my machine:** nothing of record. Every commit of mine is on origin: `cursor/na-scratch-net335-7c2e` at `1576351a9ff6`, the branch above at `4748b9dc5062`, the names at `debc13167461`, job 6 at `c299bad61230`. Three throwaway worktrees hold trial merges only; the resolution of the meeting with job 6 is a patch in my synced note's kit.

**4. Runs** [ran: `research status`, 4:33 PM PST]: `r20261113-000722-beb0` (the mended assertion, the four tests, the mirrored round) is running; a correction to my 4:09 PM line: it was launched at 4:07 PM and waited for a slot some minutes. `r20261112-213845-ddd6` (job 6's records list) is running. I report each when it ends.

